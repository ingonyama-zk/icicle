import React, { useState, useRef, useEffect } from 'react';
import { useColorMode } from '@docusaurus/theme-common';

const iconsData = {
  cpp: 'C++',
  rst: 'Rust',
  gol: 'Go',
  cpu: 'CPU',
  gpu: 'CUDA',
  app: 'Metal',
};

const iconStatus = {
  y: 'Supported',
  p: 'Partially Supported',
};

const table = [
  ['Operation', 'bn254', 'bls377', 'bls381', 'bw6', 'grumpkin', 'babybear', 'koalabear', 'stark252', 'm31', 'goldilocks', 'labrador (babybear x koalabear)'],
  ['Field Ops', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty cpuy gpuy appp'],
  ['Extension Field Ops', '', '', '', '', '', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', '', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty cpuy gpuy', ''],
  ['G1', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', '', '', '', '', '', ''],
  ['G2', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', '', '', '', '', '', '', ''],
  ['Vector Ops', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy', 'cppy rsty cpuy gpuy'],
  ['Extension Vector Ops', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy appy', '', 'cppy rsty golp cpuy gpuy appy', 'cppy rsty golp cpuy gpuy', ''],
  ['Polynomial Ops', 'cppy rstp golp cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', '', 'cppy rstp golp cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', '', 'cppy rstp golp cpuy gpuy', 'cppy cpuy gpuy'],
  ['MSM G1', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'rsty goly cpuy gpuy appy', '', '', '', '', '', ''],
  ['MSM G2', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', 'cppy rsty goly cpuy gpuy appp', '', '', '', '', '', '', ''],
  ['NTT Ops', 'cppy rstp golp cpup gpup appp', 'cppy rstp golp cpup gpup appp', 'cppy rstp golp cpup gpup appp', 'cppy rstp golp cpup gpup appp', '', 'cppy rstp golp cpup gpup appp', 'cppy rstp golp cpup gpup appp', 'cppy rstp golp cpup gpup appp', '', 'cppy rstp golp cpup gpup', 'cppy rstp cpup gpup'],
  ['FRI', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', '', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', '', 'cppy rsty cpuy gpuy', ''],
  ['ECNTT', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy goly cpuy gpuy appy', '', '', '', '', '', '', ''],
  ['Sumcheck', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', 'cppy rsty cpuy gpuy', ''],
  ['Blake3 Hash', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy', ''],
  ['Blake2s Hash', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy', ''],
  ['Keccak256 Hash', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy', ''],
  ['Keccak512 Hash', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy', ''],
  ['Sha256 Hash', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy', ''],
  ['Sha512 Hash', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy', ''],
  ['Poseidon Hash', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'rsty goly', ''],
  ['Poseidon2 Hash', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy', ''],
  ['Merkle Tree ops', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy', ''],
  ['Matrix Ops', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rstp golp cpuy gpuy appy', 'cppy rsty goly cpuy gpuy appy', 'cppy rsty goly cpuy gpuy', 'cppy rsty  cpuy gpuy'],
  ['Pairing', 'cppy rsty cpuy gpuy appy', 'cppy rsty cpuy gpuy appy', 'cppy rsty cpuy gpuy appy', 'cppy rsty cpuy gpuy appy', '', '', '', '', '', '', ''],

];

export default function FeatureSupportTable() {
  /* state */
  const [openTip, setOpenTip] = useState(null);
  const [selectedOp, setSelectedOp]   = useState('All');
  const [selectedCurve, setSelectedCurve] = useState('All');
  const [selectedTool, setSelectedTool]   = useState('All');

  /* derived lists for dropdowns */
  const operations = [...new Set(table.slice(1).map(r => r[0]))];
  const curves     = table[0].slice(1);
  const tools      = Object.keys(iconsData);

  const { colorMode } = useColorMode();
  const isDarkMode = colorMode === 'dark';
  const containerRef = useRef();

  /* click-outside to close tooltips */
  useEffect(() => {
    const cb = (e) => {
      if (!e.target.closest('[data-tooltip-target]')) setOpenTip(null);
    };
    document.addEventListener('mousedown', cb);
    return () => document.removeEventListener('mousedown', cb);
  }, []);

  /* helpers ---------------------------------------------------------- */

  const renderIcons = (cell, rowIdx, colIdx) => {
    const icons = cell
      .trim()
      .split(/\s+/)
      .filter(code => selectedTool === 'All' || code.startsWith(selectedTool));

    return (
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: 8,
        justifyItems: 'center',
        alignItems: 'center',
        height: '100%',
      }}>
        {icons.map((code, i) => {
          const m = code.match(/^(cpp|rst|gol|cpu|gpu|app)(y|p)$/);
          if (!m) return null;
          const [ , prefix, status ] = m;
          const tipId = `tip-${rowIdx}-${colIdx}-${i}`;
          const alt   = `${iconsData[prefix]} — ${iconStatus[status]}`;
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
                width={48}
                height={48}
                src={`/img/${code}.svg`}
                alt={alt}
                style={{ transition: 'transform 0.2s' }}
                onMouseOver={e => (e.currentTarget.style.transform = 'scale(1.1)')}
                onMouseOut={e  => (e.currentTarget.style.transform = 'scale(1)')}
              />
              {openTip === tipId && (
                <div style={{
                  position: 'absolute',
                  top: 52,
                  left: '50%',
                  transform: 'translateX(-50%)',
                  background: isDarkMode ? '#222' : '#333',
                  color: '#fff',
                  padding: '4px 8px',
                  borderRadius: 4,
                  fontSize: 12,
                  whiteSpace: 'nowrap',
                  zIndex: 1000,
                }}>
                  {alt}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  /* clickable curve links in header */
  const curveLinks = {
    bn254: 'https://neuromancer.sk/std/bn/bn254',
    bls377: 'https://neuromancer.sk/std/bls/BLS12-377',
    bls381: 'https://neuromancer.sk/std/bls/BLS12-381',
    bw6:    'https://eprint.iacr.org/2020/351',
    babybear: 'https://eprint.iacr.org/2023/824.pdf',
    stark252: 'https://docs.starknet.io/architecture-and-concepts/cryptography/#stark-field',
  };
  const renderHeader = (txt, i) => {
    if (i === 0) return txt;
    const key = txt.toLowerCase().replace(/[\s\-]/g, '');
    return curveLinks[key]
      ? <a href={curveLinks[key]} target="_blank" rel="noopener noreferrer"
           style={{ color: isDarkMode ? '#7cf' : '#00f', textDecoration:'underline', fontWeight:'bold' }}>
          {txt}
        </a>
      : txt;
  };

  /* ---------- JSX ---------- */

  return (
    <>
      {/* ------------- filters ------------- */}
        <div style={{
          marginBottom: 16,
          display: 'flex',
          gap: 16,
          flexWrap: 'wrap',
          alignItems: 'center'
        }}>
          <label style={{ fontWeight: 600 }}>
            Operation<br />
            <select
              value={selectedOp}
              onChange={e => setSelectedOp(e.target.value)}
              style={{
                padding: '8px 12px',
                fontSize: '16px',
                borderRadius: '8px',
                border: '1px solid #ccc',
                minWidth: '160px',
              }}
            >
              <option>All</option>
              {operations.map(op => <option key={op}>{op}</option>)}
            </select>
          </label>

          <label style={{ fontWeight: 600 }}>
            Curve<br />
            <select
              value={selectedCurve}
              onChange={e => setSelectedCurve(e.target.value)}
              style={{
                padding: '8px 12px',
                fontSize: '16px',
                borderRadius: '8px',
                border: '1px solid #ccc',
                minWidth: '160px',
              }}
            >
              <option>All</option>
              {curves.map(c => <option key={c}>{c}</option>)}
            </select>
          </label>

          <label style={{ fontWeight: 600 }}>
            Tool<br />
            <select
              value={selectedTool}
              onChange={e => setSelectedTool(e.target.value)}
              style={{
                padding: '8px 12px',
                fontSize: '16px',
                borderRadius: '8px',
                border: '1px solid #ccc',
                minWidth: '160px',
              }}
            >
              <option>All</option>
              {tools.map(t => <option key={t} value={t}>{iconsData[t]}</option>)}
            </select>
          </label>
        </div>


      {/* ------------- no-results banner ------------- */}
      {(() => {
        const visibleRows = table.slice(1).filter(row => {
          if (selectedOp !== 'All' && row[0] !== selectedOp) return false;
          return row.some((cell, colIdx) => {
            if (colIdx === 0) return false;
            if (selectedCurve !== 'All' && table[0][colIdx] !== selectedCurve) return false;
            if (!cell) return false;
            return selectedTool === 'All' || cell.includes(selectedTool);
          });
        });
        if (visibleRows.length === 0) {
          return (
            <div style={{
              marginBottom:12,
              padding:'8px 12px',
              borderRadius:4,
              border:'1px solid #a00',
              color: isDarkMode ? '#faa' : '#a00',
              background: isDarkMode ? '#222' : '#ffecec',
              fontWeight:'bold'
            }}>
              ⚠️ No results match your filter.
            </div>
          );
        }
        return null;
      })()}

      {/* ------------- table ------------- */}
      <div
        ref={containerRef}
        style={{overflow:'auto',maxHeight:'80vh',border:'1px solid #ccc',position:'relative'}}
      >
        <table style={{borderCollapse:'collapse',width:'100%'}}>
          <thead>
            <tr>
              {table[0].map((head,i)=>
                (selectedCurve==='All'||i===0||table[0][i]===selectedCurve) && (
                  <th key={i} style={{
                    position:'sticky',top:0,left:i===0?0:undefined,
                    zIndex:i===0?5:4,minWidth:i===0?'160px':'130px',
                    padding:8,textAlign:'center',
                    background:isDarkMode?'#000':'#fff',
                    color:isDarkMode?'#fff':'#000',
                    borderBottom:'1px solid #999',
                    borderRight:'1px solid #ccc'
                  }}>
                    {renderHeader(head,i)}
                  </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.slice(1).map((row,rIdx)=>{
              if (selectedOp!=='All'&&row[0]!==selectedOp) return null;
              const showRow = row.some((cell,cIdx)=>{
                if (cIdx===0) return false;
                if (selectedCurve!=='All'&&table[0][cIdx]!==selectedCurve) return false;
                if (!cell) return false;
                return selectedTool==='All'||cell.includes(selectedTool);
              });
              if (!showRow) return null;

              return (
                <tr key={rIdx}>
                  {row.map((cell,cIdx)=>{
                    if (selectedCurve!=='All'&&cIdx!==0&&table[0][cIdx]!==selectedCurve) return null;
                    return (
                      <td key={cIdx} style={{
                        padding:'6px 12px',
                        textAlign:'center',
                        verticalAlign:'middle',
                        height:110,
                        fontWeight:cIdx===0?'bold':'normal',
                        position:cIdx===0?'sticky':undefined,
                        left:cIdx===0?0:undefined,
                        background:isDarkMode?'#000':'#fff',
                        color:isDarkMode?'#fff':'#000',
                        zIndex:cIdx===0?3:1,
                        borderRight:'1px solid #eee',
                        minWidth:cIdx===0?'160px':'130px'
                      }}>
                        {cIdx===0 ? cell : renderIcons(cell,rIdx+1,cIdx)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}