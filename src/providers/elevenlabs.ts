import type { ElevenLabsClient } from "@elevenlabs/elevenlabs-js";
import {
  emptyUsage,
  type AudioFormat,
  type Provider,
  type ProviderResponse,
  type SpeakRequest,
  type SpeakResponse,
  type TranscribeRequest,
  type TranscribeResponse,
  type TranscriptWord,
  type Usage,
} from "../types";
import { CoaxUnsupportedError } from "../client";
import { EXTENSIONS } from "./audio";

export interface ElevenLabsOptions {
  /** The model of the reference: a TTS id (`eleven_*`) for speak, a Scribe id (`scribe_*`) for transcribe. */
  model: string;
  /** Required unless `client` is given. coax never falls back to the SDK's `ELEVENLABS_API_KEY` env lookup. */
  apiKey?: string;
  /** Inject an existing SDK client. Otherwise coax lazily constructs one from `apiKey` (the SDK ships
   *  inside coax and is imported only then). */
  client?: ElevenLabsClient;
  /** Host root, without `/v1` — e.g. a data-residency host. Default: the SDK's `https://api.elevenlabs.io`. */
  baseURL?: string;
  /** Headers sent with every request (per-call `headers` are merged over these). */
  headers?: Record<string, string>;
  /** Default voice id for `speak`; a per-call `voice` wins. ElevenLabs has no default voice of its own. */
  voice?: string;
}

type RequestOptions = { headers?: Record<string, string>; abortSignal?: AbortSignal };
type RawWord = { text: string; start?: number | null; end?: number | null; type: string; speakerId?: string | null };
type AnyClient = {
  textToSpeech: {
    convert(
      voiceId: string,
      body: Record<string, unknown>,
      options?: RequestOptions,
    ): { withRawResponse(): Promise<{ data: ReadableStream<Uint8Array>; rawResponse: { headers: Headers } }> };
  };
  speechToText: {
    convert(body: Record<string, unknown>, options?: RequestOptions): Promise<{ text?: unknown; words?: RawWord[]; audioDurationSecs?: number }>;
  };
};

/**
 * coax's codec → the ElevenLabs `output_format` variant. One fixed variant per codec: the API's own
 * default for mp3; Opus has one rate, at the mp3 default's bitrate; wav/pcm at 24 kHz, the highest
 * rate every tier is cleared for (44.1 kHz needs Pro). ElevenLabs offers no aac or flac.
 */
const FORMATS: Partial<Record<AudioFormat, { outputFormat: string; mediaType: string }>> = {
  mp3: { outputFormat: "mp3_44100_128", mediaType: "audio/mpeg" },
  opus: { outputFormat: "opus_48000_128", mediaType: "audio/opus" },
  wav: { outputFormat: "wav_24000", mediaType: "audio/wav" },
  pcm: { outputFormat: "pcm_24000", mediaType: "audio/pcm" },
};

/** The `character-cost` response header is what the call was billed; absent or unparsable → not reported. */
function characters(headers: Headers): number | undefined {
  const raw = headers.get("character-cost")?.trim();
  if (!raw) return undefined;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : undefined;
}

/** Spoken words only (no `spacing` / `audio_event` entries), with timing and the speaker label if any. */
function toWords(raw: RawWord[]): TranscriptWord[] {
  const out: TranscriptWord[] = [];
  for (const w of raw) {
    // A word without timing cannot be a TranscriptWord; it is still in `text`.
    if (w.type !== "word" || typeof w.start !== "number" || typeof w.end !== "number") continue;
    out.push({ text: w.text, start: w.start, end: w.end, ...(w.speakerId ? { speaker: w.speakerId } : {}) });
  }
  return out;
}

/**
 * ElevenLabs — voice only: `speak` (text-to-speech) and `transcribe` (Scribe, batch). It serves no text
 * or structured output, so those reject with CoaxUnsupportedError, as a missing optional capability does.
 */
export function elevenlabs(opts: ElevenLabsOptions): Provider {
  // The SDK reads ELEVENLABS_API_KEY when no key is passed — a key coax was not configured with must never be used.
  if (!opts.client && !opts.apiKey) throw new Error(`coax: provider "elevenlabs" needs an apiKey`);
  let client: AnyClient | undefined = opts.client as unknown as AnyClient | undefined;

  async function getClient(): Promise<AnyClient> {
    if (client) return client;
    const { ElevenLabsClient } = await import("@elevenlabs/elevenlabs-js");
    // maxRetries 0: coax's retrying() already retries transient errors; both would multiply the attempts.
    client = new ElevenLabsClient({ apiKey: opts.apiKey, maxRetries: 0, ...(opts.baseURL ? { baseUrl: opts.baseURL } : {}) }) as unknown as AnyClient;
    return client;
  }

  const requestOptions = (headers: Record<string, string> | undefined, signal: AbortSignal | undefined): RequestOptions | undefined => {
    const merged = { ...opts.headers, ...headers };
    const out: RequestOptions = {
      ...(Object.keys(merged).length ? { headers: merged } : {}),
      ...(signal ? { abortSignal: signal } : {}),
    };
    return Object.keys(out).length ? out : undefined;
  };

  return {
    name: "elevenlabs",
    model: opts.model,

    async structured(): Promise<ProviderResponse> {
      throw new CoaxUnsupportedError("structured output", "elevenlabs");
    },

    async text(): Promise<ProviderResponse> {
      throw new CoaxUnsupportedError("text generation", "elevenlabs");
    },

    async speak(req: SpeakRequest): Promise<SpeakResponse> {
      // Every check runs before the SDK is loaded: the vendor is never billed for a call coax won't honour.
      const voice = req.voice ?? opts.voice;
      if (!voice) throw new Error("coax: elevenlabs needs a voice id — pass `voice` to ai.speak() or set `voice` on the endpoint");
      if (req.instructions) throw new CoaxUnsupportedError("delivery instructions (`instructions`)", "elevenlabs");
      const format = req.format ?? "mp3";
      const variant = FORMATS[format];
      if (!variant) throw new CoaxUnsupportedError(`${format} output`, "elevenlabs");
      // The documented range; what the API does outside it (error or clamp) is not — a clamp would change the call.
      if (req.speed != null && !(req.speed >= 0.7 && req.speed <= 1.2)) {
        throw new Error(`coax: elevenlabs speed must be between 0.7 and 1.2 (got ${req.speed})`);
      }

      const c = await getClient();
      const { data, rawResponse } = await c.textToSpeech
        .convert(
          voice,
          {
            text: req.input,
            modelId: opts.model,
            outputFormat: variant.outputFormat,
            ...(req.language ? { languageCode: req.language } : {}),
            ...(req.speed != null ? { voiceSettings: { speed: req.speed } } : {}),
          },
          requestOptions(req.headers, req.signal),
        )
        .withRawResponse();
      const audio = new Uint8Array(await new Response(data).arrayBuffer());
      // Never an empty success: no audio is a failure, whatever the status said.
      if (audio.byteLength === 0) throw new Error("coax: elevenlabs returned no audio");
      const billed = characters(rawResponse.headers);
      const usage: Usage = { ...emptyUsage(), ...(billed != null ? { characters: billed } : {}) };
      return { audio, mediaType: variant.mediaType, usage, model: opts.model };
    },

    async transcribe(req: TranscribeRequest): Promise<TranscribeResponse> {
      // ElevenLabs has no free-text context; keyterms/transcript_edit mean (and cost) something else.
      if (req.prompt) throw new CoaxUnsupportedError("a transcription prompt (`prompt`)", "elevenlabs");
      const { mediaType } = req.audio;
      const filename = req.audio.filename ?? (mediaType ? `audio.${EXTENSIONS[mediaType] ?? "wav"}` : "audio");
      const c = await getClient();
      const res = await c.speechToText.convert(
        {
          modelId: opts.model,
          file: { data: req.audio.data, filename, ...(mediaType ? { contentType: mediaType } : {}) },
          ...(req.language ? { languageCode: req.language } : {}),
          ...(req.speakers ? { diarize: true } : {}),
          // Keep "(laughter)"-style tags out of `text`, so it holds what was said — as on every other vendor.
          tagAudioEvents: false,
        },
        requestOptions(req.headers, req.signal),
      );
      // The multichannel and webhook shapes (never requested by coax) carry no top-level text.
      if (typeof res.text !== "string") throw new Error("coax: elevenlabs returned no transcript text");
      const words = Array.isArray(res.words) ? toWords(res.words) : undefined;
      const seconds = res.audioDurationSecs;
      const usage: Usage = { ...emptyUsage(), ...(typeof seconds === "number" && Number.isFinite(seconds) ? { audioSeconds: seconds } : {}) };
      return { text: res.text, ...(words ? { words } : {}), usage, model: opts.model };
    },
  };
}
