import { Validator, format } from '@cfworker/json-schema';
import type { Schema } from '@cfworker/json-schema';
import meta from './json-schema-meta.json';
import { ApiError } from './api-error';

const cache = new Map<string, Validator>();
const meta2020 = new Validator(meta.draft2020 as Schema, '2020-12', false);
for (const vocabulary of meta.vocabularies) meta2020.addSchema(vocabulary as unknown as Schema);
const meta7 = new Validator(meta.draft7 as Schema, '7', false);

function compile(input: Record<string, unknown>): Validator {
  const encoded = JSON.stringify(input);
  if (new TextEncoder().encode(encoded).byteLength > 65536) throw new ApiError(400, 'Submission schema must be at most 64 KiB');
  const existing = cache.get(encoded);
  if (existing) return existing;
  const schema = JSON.parse(encoded) as Schema;
  const draft = schema.$schema;
  if (draft && !['https://json-schema.org/draft/2020-12/schema', 'https://json-schema.org/draft/2020-12/schema#',
    'http://json-schema.org/draft-07/schema#', 'https://json-schema.org/draft-07/schema#'].includes(draft)) {
    throw new ApiError(400, 'Use JSON Schema draft 2020-12 or draft-07');
  }
  const references: string[] = [];
  const anchors = new Set<string>();
  function inspect(value: unknown, depth = 0): void {
    if (depth > 64) throw new ApiError(400, 'Submission schema exceeds the nesting limit of 64');
    if (!value || typeof value !== 'object') return;
    if (Array.isArray(value)) { value.forEach((child) => inspect(child, depth + 1)); return; }
    for (const [key, child] of Object.entries(value)) {
      if (['__proto__', '__absolute_ref__', '__absolute_uri__', '__absolute_recursive_ref__'].includes(key)) {
        throw new ApiError(400, 'Submission schema contains an unsupported property');
      }
      if (key === '$async' || key === '$dynamicRef' || key === '$recursiveRef') {
        throw new ApiError(400, 'Hosted submission schemas use synchronous validation and static local $ref references');
      }
      if (key === '$ref') {
        if (typeof child !== 'string' || !child.startsWith('#')) throw new ApiError(400, 'Submission schema references must be local fragments');
        references.push(child);
      }
      if (key === '$anchor' && typeof child === 'string') anchors.add(child);
      if (key === 'format' && (typeof child !== 'string' || !Object.hasOwn(format, child))) {
        throw new ApiError(400, `Unsupported submission schema format: ${String(child)}`);
      }
      if (key === 'pattern' && typeof child === 'string') {
        try { new RegExp(child, 'u'); } catch { throw new ApiError(400, 'Submission schema contains an invalid regular expression'); }
      }
      inspect(child, depth + 1);
    }
  }
  inspect(schema);
  const validation = (draft?.includes('draft-07') ? meta7 : meta2020).validate(schema);
  if (!validation.valid) throw new ApiError(400, `Invalid submission schema: ${validation.errors.slice(0, 5).map((error) => `${error.instanceLocation}: ${error.error}`).join('; ')}`);
  for (const reference of references) {
    if (reference === '#') continue;
    if (!reference.startsWith('#/')) {
      if (!anchors.has(reference.slice(1))) throw new ApiError(400, `Submission schema reference ${reference} was not found`);
      continue;
    }
    let current: unknown = schema;
    try {
      for (const segment of decodeURIComponent(reference.slice(2)).split('/').map((part) => part.replaceAll('~1', '/').replaceAll('~0', '~'))) {
        if (!current || typeof current !== 'object' || !Object.hasOwn(current, segment)) throw new Error();
        current = (current as Record<string, unknown>)[segment];
      }
      if (typeof current !== 'boolean' && (!current || typeof current !== 'object' || Array.isArray(current))) throw new Error();
    } catch { throw new ApiError(400, `Submission schema reference ${reference} was not found`); }
  }
  let validator: Validator;
  try { validator = new Validator(schema, draft?.includes('draft-07') ? '7' : '2020-12', false); }
  catch { throw new ApiError(400, 'Invalid submission schema references'); }
  if (cache.size >= 64) cache.delete(cache.keys().next().value!);
  cache.set(encoded, validator);
  return validator;
}

export function validateSubmissionSchema(schema: Record<string, unknown>): void { compile(schema); }
export function validateSubmissionPayload(schema: Record<string, unknown>, payload: unknown): void {
  let result;
  try { result = compile(schema).validate(payload); }
  catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(422, 'Submission could not be validated against the benchmark format');
  }
  if (!result.valid) throw new ApiError(422, `Submission does not match the benchmark format. ${result.errors.slice(0, 8)
    .map((error) => `payload${error.instanceLocation.replace(/^#/, '')}: ${error.error}`).join('; ')}`);
}
