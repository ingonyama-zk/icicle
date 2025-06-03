import React from 'react';
import Slider from 'react-slick';
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import { useColorMode } from '@docusaurus/theme-common';

// The slides data (base names only)
const slides = [
  {
    baseImg: 'FRI_graph_1',
  },
  {
    baseImg: 'MSM_graph_1',
  },
  {
    baseImg: 'SHA3_graph_1',
  },
  {
    baseImg: 'Sumcheck_graph_1',
  }
];

// Custom arrows with dynamic color
function NextArrow({ onClick, color }) {
  return (
    <div
      style={{
        position: 'absolute',
        top: '50%',
        right: '-40px',
        transform: 'translateY(-50%)',
        zIndex: 1,
        cursor: 'pointer',
        fontSize: '24px',
        color: color
      }}
      onClick={onClick}
    >
      ➔
    </div>
  );
}

function PrevArrow({ onClick, color }) {
  return (
    <div
      style={{
        position: 'absolute',
        top: '50%',
        left: '-40px',
        transform: 'translateY(-50%)',
        zIndex: 1,
        cursor: 'pointer',
        fontSize: '24px',
        color: color
      }}
      onClick={onClick}
    >
      ←
    </div>
  );
}

// Main carousel component
export default function BenchmarkCarousel() {
  const { colorMode } = useColorMode();
  const isDarkMode = colorMode === 'dark';
  const arrowColor = isDarkMode ? '#eee' : '#333';

  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    fade: true, // <--- THE MAIN CHANGE
    slidesToShow: 1,
    slidesToScroll: 1,
    arrows: true,
    nextArrow: <NextArrow color={arrowColor} />,
    prevArrow: <PrevArrow color={arrowColor} />
  };

  const getImageFilename = (baseImg) => {
    return isDarkMode 
      ? `/img/${baseImg.replace('_', 'drk_')}.svg`
      : `/img/${baseImg}.png`;
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem 0', position: 'relative' }}>
      <Slider {...settings}>
        {slides.map((slide, index) => {
          const imageFilename = getImageFilename(slide.baseImg);
          return (
            <div key={index} style={{ textAlign: 'center' }}>
              <img 
                src={imageFilename} 
                alt={slide.title} 
                style={{ 
                  display: 'block',
                  margin: '0 auto',
                  width: '100%',
                  maxWidth: '1000px',
                  height: 'auto',
                  maxHeight: '800px'
                }}
              />
              <h3 style={{ marginTop: '1rem' }}>{slide.title}</h3>
              <p style={{ color: isDarkMode ? '#aaa' : '#666' }}>{slide.subtitle}</p>
            </div>
          );
        })}
      </Slider>
    </div>
  );
}
