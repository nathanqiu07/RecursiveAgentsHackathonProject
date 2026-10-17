import { useRef, useState, type KeyboardEvent } from 'react';
import Resources from './Resources';
import Benchmarks from './Benchmarks';
import Guide from './Guide';
import Agents from './Agents';
import Connections from './Connections';
import DataCenter from './DataCenter';

const tabs = ['Benchmarks', 'AI Data Center', 'Library', 'Agents', 'Connections', 'Resources', 'Guide'] as const;
const slug = (tab: string) => tab.toLowerCase().replace(/\s+/g, '-');

export default function App() {
  const [activeTab, setActiveTab] = useState<(typeof tabs)[number]>('Benchmarks');
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let nextIndex = index;

    switch (event.key) {
      case 'ArrowRight':
        nextIndex = (index + 1) % tabs.length;
        break;
      case 'ArrowLeft':
        nextIndex = (index - 1 + tabs.length) % tabs.length;
        break;
      case 'Home':
        nextIndex = 0;
        break;
      case 'End':
        nextIndex = tabs.length - 1;
        break;
      default:
        return;
    }

    event.preventDefault();
    setActiveTab(tabs[nextIndex]);
    tabRefs.current[nextIndex]?.focus();
  }

  return (
    <div className="dashboard">
      <div className="tabs" role="tablist" aria-label="Dashboard">
        {tabs.map((tab, index) => (
          <button
            key={tab}
            ref={(element) => {
              tabRefs.current[index] = element;
            }}
            className="tab"
            id={`${slug(tab)}-tab`}
            type="button"
            role="tab"
            aria-selected={activeTab === tab}
            aria-controls={`${slug(tab)}-panel`}
            tabIndex={activeTab === tab ? 0 : -1}
            onClick={() => setActiveTab(tab)}
            onKeyDown={(event) => handleKeyDown(event, index)}
          >
            {tab}
          </button>
        ))}
      </div>

      <main className="content">
        {tabs.map((tab) => (
          <section
            key={tab}
            id={`${slug(tab)}-panel`}
            role="tabpanel"
            aria-labelledby={`${slug(tab)}-tab`}
            hidden={activeTab !== tab}
            tabIndex={0}
          >
            {tab === 'Benchmarks' && <Benchmarks active={activeTab === 'Benchmarks'} />}
            {tab === 'AI Data Center' && <DataCenter active={activeTab === 'AI Data Center'} />}
            {tab === 'Resources' && <Resources active={activeTab === 'Resources'} />}
            {tab === 'Agents' && <Agents active={activeTab === 'Agents'} />}
            {tab === 'Connections' && <Connections active={activeTab === 'Connections'} />}
            {tab === 'Guide' && <Guide />}
          </section>
        ))}
      </main>
    </div>
  );
}
