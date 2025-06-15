import React from 'react';
import { useColorMode } from '@docusaurus/theme-common';
import { useLocation, useHistory } from '@docusaurus/router';

const benchmarks = [
  { title: 'FRI', baseImg: 'FRI_graph_1' },
  { title: 'MSM', baseImg: 'MSM_graph_1' },
  { title: 'PQC', baseImg: 'pqc_graph_1' },
  { title: 'SHA3', baseImg: 'SHA3_graph_1' },
  { title: 'Sumcheck', baseImg: 'Sumcheck_graph_1' },
];

export default function BenchmarkGallery() {
  const { colorMode } = useColorMode();
  const isDarkMode = colorMode === 'dark';
  const location = useLocation();
  const history = useHistory();

  const search = new URLSearchParams(location.search);
  const view = search.get('view');
  const active = benchmarks.find(b => b.title.toLowerCase() === view?.toLowerCase());

  const getImagePath = (baseImg) => {
    const darkName = baseImg.replace('_', 'drk_');
    return isDarkMode
      ? `/img/darkmode/${darkName}.png`
      : `/img/BenchmarkImages/${baseImg}.png`;
  };

  const openModal = (title) => {
    history.push(`${location.pathname}?view=${title.toLowerCase()}`);
  };

  const closeModal = () => {
    history.push(location.pathname.split('?')[0]);
  };

  return (
    <div>
     <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '2rem',
        padding: '2rem 0'
      }}>
        {benchmarks.map(({ title, baseImg }) => (
          <div key={baseImg} style={{ textAlign: 'center' }}>
            <div onClick={() => openModal(title)} style={{ cursor: 'pointer' }}>
              <img
                src={getImagePath(baseImg)}
                alt={title}
                style={{
                  maxWidth: '100%',
                  height: 'auto',
                  borderRadius: '8px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}
              />
            </div>
            <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>{title}</div>
          </div>
        ))}
      </div>

          {active && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0,
          width: '100%',
          height: '100%',
          background: 'rgba(0,0,0,0.7)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 9999,
        }}>
          <div style={{
            background: '#fff',
            color: '#000',
            padding: '0',
            borderRadius: '10px',
            maxWidth: '95%',
            maxHeight: '95%',
            overflow: 'auto',
            position: 'relative'
          }}>
            <button onClick={closeModal} style={{
              position: 'absolute',
              top: '1rem',
              right: '1rem',
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              cursor: 'pointer',
              fontWeight: 'bold',
              zIndex: 1
            }}>×</button>

            <img
              src={getImagePath(active.baseImg)}
              alt={active.title}
              style={{
                display: 'block',
                maxWidth: '100%',
                maxHeight: '100vh',
                objectFit: 'contain',
                margin: '0 auto',
                borderRadius: '8px',
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
