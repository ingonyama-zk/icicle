import React, { useState, useEffect, useRef } from 'react';
import Slider from 'react-slick';
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import { useColorMode } from '@docusaurus/theme-common';

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth <= 768);
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return isMobile;
}

// Slides data
const slides = [
  { baseImg: 'FRI_graph_1'},
  { baseImg: 'MSM_graph_1'},
  { baseImg: 'SHA3_graph_1'},
  { baseImg: 'Sumcheck_graph_1'},
];

export default function BenchmarkCarousel() {
  const { colorMode } = useColorMode();
  const isDarkMode = colorMode === 'dark';
  const isMobile = useIsMobile();
  const sliderRef = useRef(null);

  const getImageFilename = (baseImg) => {
    if (isDarkMode) {
      const darkFilename = baseImg.replace('_', 'drk_');
      return `/img/darkmode/${darkFilename}.png`;
    } else {
      return `/img/${baseImg}.png`;
    }
  };

  const settings = {
    dots: false,  // remove dots
    infinite: true,
    speed: 500,
    fade: true,
    slidesToShow: 1,
    slidesToScroll: 1,
    swipe: true,
    arrows: false,  // fully disable slick arrows
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem 0', position: 'relative' }}>
      <Slider ref={sliderRef} {...settings}>
        {slides.map((slide, index) => {
          const imageFilename = getImageFilename(slide.baseImg);
          return (
            <div key={index} style={{ textAlign: 'center' }}>
              <img 
                src={imageFilename} 
                loading="lazy"
                style={{ 
                  display: 'block',
                  margin: '0 auto',
                  width: '100%',
                  maxWidth: '1000px',
                  height: 'auto',
                  maxHeight: '800px'
                }}
              />
            </div>
          );
        })}
      </Slider>

      {/* Custom arrows under the image */}
      <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center', gap: '2rem' }}>
        <div onClick={() => sliderRef.current?.slickPrev()} style={{ cursor: 'pointer' }}>
          <img 
            src={isDarkMode ? '/img/darkmode/warl.svg' : '/img/barl.svg'} 
            alt="Previous" 
            style={{ width: '40px', height: '40px' }} 
          />
        </div>
        <div onClick={() => sliderRef.current?.slickNext()} style={{ cursor: 'pointer' }}>
          <img 
            src={isDarkMode ? '/img/darkmode/warr.svg' : '/img/barr.svg'} 
            alt="Next" 
            style={{ width: '40px', height: '40px' }} 
          />
        </div>
      </div>
    </div>
  );
}
