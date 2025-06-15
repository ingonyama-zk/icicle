import React from 'react';

const items = [
  { href: "/api/cpp/MSM", label: "MSM" },
  { href: "/api/cpp/NTT", label: "NTT / ECNTT" },
  { href: "/api/cpp/Vector-operations", label: "Vector Operations" },
  { href: "/api/cpp/Program", label: "Program" },
  { href: "/api/cpp/Polynomials/overview", label: "Polynomials" },
  { href: "/api/cpp/Hash", label: "Hash" },
  { href: "/api/cpp/Merkle-Tree", label: "Merkle Tree" },
  { href: "/api/cpp/Sumcheck", label: "Sumcheck" },
  { href: "/api/cpp/FRI", label: "FRI" },
  { href: "/api/cpp/Pairings", label: "Pairings" },
  { href: "/api/cpp/Serialization", label: "Serialization" },
];

export default function CustomGrid() {
  return (
    <div style={{
      display: 'grid',
      gridTemplateAreas: `
        ". a b c ."
        "d e f g h "
        ". i j k ."
      `,
      gap: '1rem',
      justifyContent: 'center',
      marginTop: '1rem'
    }}>
      {items.map((item, index) => (
        <a
          key={index}
          href={item.href}
          className="card-link"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1rem',
            background: '#f9f9f9',
            borderRadius: '8px',
            textDecoration: 'none',
            color: '#333',
            fontWeight: '500',
            boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
            transition: 'transform 0.2s, box-shadow 0.2s',
            gridArea: getGridArea(index)
          }}
          onMouseEnter={e => {
            e.currentTarget.style.transform = 'translateY(-4px)';
            e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
          }}
          onMouseLeave={e => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 2px 6px rgba(0,0,0,0.1)';
          }}
        >
          {item.label}
        </a>
      ))}
    </div>
  );
}

function getGridArea(index) {
  const areas = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'];
  return areas[index];
}
