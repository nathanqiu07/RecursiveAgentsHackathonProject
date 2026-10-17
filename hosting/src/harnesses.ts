export type HarnessId = 'builtin' | 'codex' | 'claude-code' | 'nemoclaw';
export type HarnessModel = { id: string; label?: string };
export type HarnessInfo = { id: HarnessId; label: string; available: boolean; message: string; models: HarnessModel[] };
export type ModelProfile = { id: string; name: string; harness: HarnessId; model: string; base_url: string | null; api_key_set: boolean; auth_mode: 'codex-oauth' | 'api-key' | 'environment'; enabled: boolean };
export type HarnessCatalog = { harnesses: HarnessInfo[]; profiles: ModelProfile[]; default_harness: HarnessId };

export const harnessLabel = (id: HarnessId | undefined) => id === 'codex' ? 'Codex' : id === 'claude-code' ? 'Claude Code' : id === 'nemoclaw' ? 'NVIDIA NemoClaw' : 'Built-in';
