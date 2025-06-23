import React, { useState, useRef, useEffect } from 'react';
import { useColorMode } from '@docusaurus/theme-common';

const iconsData = {
  cpp: 'C++',
  rst: 'Rust',
  gol: 'Go',
  cpu: 'CPU',
  gpu: 'CUDA',
  app: 'Metal'
};

const iconStatus = {
  y: 'Supported',
  p: 'Partially Supported'
};

const table = [
  ['Operation', 'bn254', 'bls377', 'bls381', 'bw6', 'grumpkin', 'babybear', 'koalabear', 'stark252', 'm31', 'goldilocks', 'labrador (babybear x koalabear)'],
  ['Field Ops', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty cpuy gpuy appp'],
  ['Extension Field Ops', '', '', '', '', '', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', '', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty cpuy gpuy', ''],
  ['G1', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', '', '', '', '', '', ''],
  ['G2', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'rsty golp', '', '', '', '', '', ''],
  ['Vector Ops', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy', 'cppy rsty cpuy gpuy'],
  ['Extension Vector Ops', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'golp', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy', ''],
  ['Polynomial Ops', 'cppy rstp golp cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', 'rstp golp', 'cppy rstp golp cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', 'rstp golp', 'cppy rstp golp cpuy gpuy', 'cppy cpuy gpuy'],
  ['MSM G1', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'rsty goly cpuy gpuy appy', '', '', '', '', '', ''],
  ['MSM G2', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'rsty goly', '', '', '', '', '', ''],
  ['NTT Ops', 'cppy rstp golp cpup gpup appp', 'cppy rstp golp cpup gpup appp', 'cppy rstp golp cpup gpup appp', 'cppy rstp golp cpup gpup appp', '', 'cppy rstp golp cpup gpup appp', 'cppy rstp golp cpup gpup appp', 'cppy rstp golp cpup gpup appp', '', 'cppy rstp golp cpup gpup', 'cppy rstp cpup gpup'],
  ['FRI', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', '', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', '', 'cppy rsty cpuy gpuy', 'cppy'],
  ['ECNTT', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy goly cpuy gpuy appy', 'cppy', '', '', '', '', '', ''],
  ['Sumcheck', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'rsty'],
  ['Blake3 Hash', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy', 'rsty']
];

export default function FeatureSupportTable() {
  const [openTip, setOpenTip] = useState(null);
  const containerRef = useRef();
  const { colorMode } = useColorMode();
  const isDarkMode = colorMode === 'dark';

  useEffect(() => {
    const handleClick = (e) => {
      const clickedInsideTooltip = e.target.closest('[data-tooltip-target]');
      if (!clickedInsideTooltip) {
        setOpenTip(null);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const renderIcons = (cell, rowIdx, colIdx) => {
    const icons = cell.trim().split(/\s+/);
    return (
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: '8px',
        justifyItems: 'center',
        alignItems: 'center',
        height: '100%',
      }}>
        {icons.map((iconCode, i) => {
          const match = iconCode.match(/^(cpp|rst|gol|cpu|gpu|app)(y|p)$/);
          if (!match) return null;
          const [_, prefix, status] = match;
          const alt = `${iconsData[prefix]} — ${iconStatus[status]}`;
          const tipId = `tip-${rowIdx}-${colIdx}-${i}`;
          return (
            <div
              key={tipId}
              data-tooltip-target
              style={{ position: 'relative', cursor: 'pointer' }}
              onClick={(e) => {
                e.stopPropagation();
                setOpenTip(openTip === tipId ? null : tipId);
              }}
            >
              <img
                src={`/img/${iconCode}.svg`}
                alt={alt}
                width="48"
                height="48"
                style={{ transition: 'transform 0.2s ease' }}
                onMouseOver={(e) => (e.currentTarget.style.transform = 'scale(1.1)')}
                onMouseOut={(e) => (e.currentTarget.style.transform = 'scale(1)')}
              />
              {openTip === tipId && (
                <div
                  data-tooltip-target
                  style={{
                    position: 'absolute',
                    top: '52px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    background: isDarkMode ? '#222' : '#333',
                    color: '#fff',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '12px',
                    whiteSpace: 'nowrap',
                    zIndex: 1000,
                  }}
                >
                  {alt}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  const curveLinks = {
    bn254: 'https://neuromancer.sk/std/bn/bn254',
    bls377: 'https://neuromancer.sk/std/bls/BLS12-377',
    bls381: 'https://neuromancer.sk/std/bls/BLS12-381',
    bw6: 'https://eprint.iacr.org/2020/351',
    babybear: 'https://eprint.iacr.org/2023/824.pdf',
    stark252: 'https://docs.starknet.io/architecture-and-concepts/cryptography/#stark-field',
  };

  const renderHeader = (text, i) => {
    if (i === 0) return text;
    const normalized = text.toLowerCase().replace(/[\s\-]/g, '');
    const link = curveLinks[normalized];
    return link ? (
      <a
        href={link}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          color: isDarkMode ? '#7cf' : '#00f',
          textDecoration: 'underline',
          fontWeight: 'bold',
        }}
      >
        {text}
      </a>
    ) : text;
  };

  return (
        <div
        ref={containerRef}
        style={{
            overflowX: 'auto',
            overflowY: 'auto',    
            maxHeight: '80vh',
            position: 'relative',  
            border: '1px solid #ccc',
        }}
        >
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <thead>
          <tr>
            {table[0].map((header, i) => (
              <th
                key={i}
                style={{
                  position: 'sticky',
                  top: 0, // Sticks to the top of the scrollable container (the parent div)
                  left: i === 0 ? 0 : undefined, // Sticks the first column header to the left
                  zIndex: i === 0 ? 5 : 4, // Z-index for headers, sticky corner needs highest
                  minWidth: i === 0 ? '160px' : '130px', // Maintain widths
                  padding: '8px',
                  textAlign: 'center',
                  background: isDarkMode ? '#000' : '#fff', // Crucial for sticky elements to not be transparent
                  color: isDarkMode ? '#fff' : '#000',
                  borderBottom: '1px solid #999',
                  borderRight: '1px solid #ccc',
                }}
              >
                {renderHeader(header, i)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {/* Iterate over ALL data rows, including the first one (Field Ops, etc.) */}
          {table.slice(1).map((row, rowIdxOffset) => ( // Changed rowIdx to rowIdxOffset for clarity
            <tr key={rowIdxOffset}>
              {row.map((cell, colIdx) => (
                <td
                  key={colIdx}
                  style={{
                    padding: '6px 12px',
                    verticalAlign: 'middle',
                    textAlign: 'center',
                    height: '110px',
                    fontWeight: colIdx === 0 ? 'bold' : 'normal',
                    position: colIdx === 0 ? 'sticky' : undefined, // Make the first column sticky
                    left: colIdx === 0 ? 0 : undefined, // Sticks to the left of the scrollable container
                    background: isDarkMode ? '#000' : '#fff', // Crucial for sticky elements
                    color: isDarkMode ? '#fff' : '#000',
                    zIndex: colIdx === 0 ? 3 : 1, // Z-index for sticky first column, below sticky header
                    borderRight: '1px solid #eee',
                    minWidth: colIdx === 0 ? '160px' : '130px', // Maintain widths for cells
                  }}
                >
                  {/* Correctly pass the actual row index for tooltip ID */}
                  {colIdx === 0 ? cell : renderIcons(cell, rowIdxOffset + 1, colIdx)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}