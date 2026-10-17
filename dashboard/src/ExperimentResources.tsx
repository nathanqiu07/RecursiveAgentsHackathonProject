import { useEffect, useState } from 'react';

type Pool = {
  id: string; node_id: string; description: string;
  status: 'available' | 'busy' | 'offline' | 'unavailable' | 'unregistered';
  available_slots: number; total_slots: number; held_jobs: number;
  available_memory_mb: number; memory_mb: number;
};
type Snapshot = { resources: Pool[]; checked_at: number; source: 'resource_manager' };
const labels = { available: 'Available', busy: 'Busy', offline: 'Offline', unavailable: 'Unreachable', unregistered: 'Not registered' };

export default function ExperimentResources({ active, refreshToken }: { active: boolean; refreshToken: number }) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer: number | undefined;
    async function poll() {
      try {
        const response = await fetch('/api/agents/resource-status', { cache: 'no-store', signal: controller.signal });
        if (!response.ok) throw new Error('Shared resource availability could not be checked.');
        const value = await response.json() as Snapshot;
        if (!Array.isArray(value.resources) || value.source !== 'resource_manager') throw new Error('Shared resources returned an unreadable response.');
        if (!controller.signal.aborted) { setSnapshot(value); setError(''); }
      } catch (failure) {
        if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : 'Shared resource availability could not be checked.');
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(() => { void poll(); }, 5000);
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [active, refreshToken]);

  const pools = snapshot?.resources || [];
  return <section className="experiment-resources" aria-labelledby="experiment-resources-title">
    <div className="section-heading"><h2 id="experiment-resources-title">Experiment resources</h2><span className="agent-count">{error ? 'Availability unknown' : snapshot ? `${pools.filter((pool) => pool.status === 'available').length} available · ${pools.filter((pool) => pool.status === 'busy').length} busy` : 'Checking…'}</span></div>
    <p className="agent-field-note">Experiments and benchmark verification share these resources. Each GPU node runs one VM job at a time; busy nodes become available after cleanup.</p>
    {error && <p className="agent-warning" role="status">{error}{snapshot && ' Showing the last known snapshot.'}</p>}
    {snapshot && !pools.length && <p className="muted agent-detail-copy">No resource nodes have been enrolled yet. Add the A5000 servers to Resource Manager to make them available here.</p>}
    {!!pools.length && <div className="experiment-resource-list">{pools.map((pool) => <div className="experiment-resource-row" key={pool.id}>
      <div><strong>{pool.id}</strong><span>{pool.node_id}{pool.description && ` · ${pool.description}`}</span></div>
      <div><span className={`agent-status agent-status-${error ? 'needs_attention' : pool.status === 'available' ? 'completed' : pool.status === 'busy' ? 'running' : 'needs_attention'}`}>{error ? 'Unknown' : labels[pool.status]}</span><span>{pool.available_slots}/{pool.total_slots} slots free · {(pool.available_memory_mb / 1024).toFixed(1)} GiB RAM unreserved</span></div>
    </div>)}</div>}
  </section>;
}
