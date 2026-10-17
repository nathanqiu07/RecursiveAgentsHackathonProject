export type MiniQualityBaseline = {
  mmlu: number; gsm8k: number; ppl: number; identity: string;
  mmlu_correct?: number; mmlu_count?: number;
  gsm8k_correct?: number; gsm8k_count?: number;
};

export type NemotronVerifierStatus = {
  configured: boolean; reachable: boolean; ready: boolean; baseline_ready: boolean;
  baseline: MiniQualityBaseline | null; message: string;
  baseline_status?: string;
  configured_nodes?: number; online_nodes?: number;
  node_baselines?: { node_id: string; status: string; metrics: MiniQualityBaseline | null }[];
};
