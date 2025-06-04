import React, { useState, useEffect } from 'react';
import Slider from 'react-slick';
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import { useColorMode } from '@docusaurus/theme-common';

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    function handleResize() {
      setIsMobile(window.innerWidth <= 768);
    }

    handleResize(); // Call once on mount

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return isMobile;
}

// The slides data
const slides = [
  { baseImg: 'FRI_graph_1' },
  { baseImg: 'MSM_graph_1' },
  { baseImg: 'SHA3_graph_1' },
  { baseImg: 'Sumcheck_graph_1' },
];

// Custom arrows with dynamic SVG
function NextArrow({ onClick, isDarkMode, isMobile }) {
  if (isMobile) return null;
  const arrow = isDarkMode ? '/img/darkmode/warr.svg' : '/img/barr.svg';
  return (
    <div
      style={{
        position: 'absolute',
        top: '50%',
        right: '-5%',
        transform: 'translateY(-50%)',
        zIndex: 1,
        cursor: 'pointer',
      }}
      onClick={onClick}
    >
      <img src={arrow} alt="Next" style={{ width: '40px', height: '40px' }} />
    </div>
  );
}

function PrevArrow({ onClick, isDarkMode, isMobile }) {
  if (isMobile) return null;
  const arrow = isDarkMode ? '/img/darkmode/warl.svg' : '/img/barl.svg';
  return (
    <div
      style={{
        position: 'absolute',
        top: '50%',
        left: '-5%',
        transform: 'translateY(-50%)',
        zIndex: 1,
        cursor: 'pointer',
      }}
      onClick={onClick}
    >
      <img src={arrow} alt="Previous" style={{ width: '40px', height: '40px' }} />
    </div>
  );
}

export default function BenchmarkCarousel() {
  const { colorMode } = useColorMode();
  const isDarkMode = colorMode === 'dark';
  const isMobile = useIsMobile();

  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    fade: true,
    slidesToShow: 1,
    slidesToScroll: 1,
    swipe: true,
    arrows: !isMobile,
    nextArrow: <NextArrow isDarkMode={isDarkMode} isMobile={isMobile} />,
    prevArrow: <PrevArrow isDarkMode={isDarkMode} isMobile={isMobile} />,
    appendDots: dots => (
      <div style={{ bottom: '-30px' }}>
        <ul style={{ margin: '0px' }}>{dots}</ul>
      </div>
    ),
    customPaging: i => (
      <div
        style={{
          width: '12px',
          height: '12px',
          borderRadius: '50%',
          background: isDarkMode ? '#bbb' : '#999',
          display: 'inline-block',
        }}
      />
    ),
  };

  const getImageFilename = (baseImg) => {
    return isDarkMode 
      ? `/img/${baseImg.replace('_', 'drk_')}.png`
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
