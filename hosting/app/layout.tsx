import type { Metadata } from 'next';
import '../src/styles.css';
import '../src/benchmarks.css';

export const metadata: Metadata = {
  title: 'Crusoe Benchmark Dashboard',
  description: 'Open-ended benchmarks, full submission contracts, and independently verified leaderboards.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
