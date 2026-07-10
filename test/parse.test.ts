import { describe, expect, it } from "vitest";
import { extractJson } from "../src/parse";

describe("extractJson", () => {
  it("passes an object through untouched (native structured mode)", () => {
    const o = { a: 1, b: [2, 3] };
    expect(extractJson(o)).toBe(o);
  });

  it("parses clean JSON", () => {
    expect(extractJson('{"a":1}')).toEqual({ a: 1 });
  });

  it("strips a ```json fence", () => {
    expect(extractJson('```json\n{"a": 1}\n```')).toEqual({ a: 1 });
    expect(extractJson('```\n{"a": 1}\n```')).toEqual({ a: 1 });
  });

  it("repairs malformed JSON (trailing comma, single quotes, unquoted key)", () => {
    expect(extractJson("{'a': 1, b: 2,}")).toEqual({ a: 1, b: 2 });
  });

  it("repairs a truncated object", () => {
    expect(extractJson('{"a": 1, "b": "hello')).toEqual({ a: 1, b: "hello" });
  });

  it("returns the original string when nothing parses", () => {
    expect(extractJson("not json at all")).toBe("not json at all");
  });
});
