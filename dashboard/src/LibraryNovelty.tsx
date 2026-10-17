export type LibraryNoveltyReview = {
  verdict: 'novel' | 'not_novel' | 'inconclusive';
  rationale: string;
  source_paper_ids: string[];
  snapshot_fingerprint: string;
  snapshot_count: number;
  reviewed_at: string;
  method: 'semantic' | 'exact_duplicate' | 'precheck';
};

const verdictLabels = { novel: 'Novel in checked Library', not_novel: 'Not novel', inconclusive: 'Inconclusive' };
const methodLabels = { semantic: 'Semantic comparison', exact_duplicate: 'Exact duplicate check', precheck: 'Precheck' };

export default function LibraryNovelty({ review }: { review: LibraryNoveltyReview | null | undefined }) {
  const reviewed = review ? new Date(review.reviewed_at) : null;
  const validDate = reviewed && Number.isFinite(reviewed.getTime()) ? reviewed : null;
  return <div className="agent-library-review" aria-labelledby="independent-library-novelty-title">
    <div className="agent-library-review-heading"><h4 id="independent-library-novelty-title">Independent Library novelty</h4>
      {review ? <span className={`agent-status agent-status-${review.verdict}`}>{verdictLabels[review.verdict]}</span> : <span className="agent-status">Not recorded</span>}
    </div>
    {review ? <>
      <p className="agent-detail-copy">{review.rationale}</p>
      <p className="agent-field-note">The checked Library snapshot contains {review.snapshot_count.toLocaleString()} other {review.snapshot_count === 1 ? 'paper' : 'papers'}, including pending submissions. This verdict applies to that saved snapshot.</p>
      <details className="agent-disclosure agent-library-snapshot"><summary>Source papers and checked snapshot</summary>
        {review.source_paper_ids.length ? <><p>Source papers cited by the Library reviewer:</p><ul>{review.source_paper_ids.map((id) => <li key={id}><code>{id}</code></li>)}</ul></> : <p>No source paper IDs were cited.</p>}
        <dl><div><dt>Review method</dt><dd>{methodLabels[review.method]}</dd></div><div><dt>Reviewed</dt><dd>{validDate ? <time dateTime={validDate.toISOString()}>{validDate.toLocaleString()}</time> : 'Review time unavailable'}</dd></div><div><dt>Snapshot fingerprint</dt><dd><code>{review.snapshot_fingerprint}</code></dd></div></dl>
      </details>
    </> : <p className="agent-field-note">The Library has not recorded an independent novelty verdict for this submission.</p>}
  </div>;
}
