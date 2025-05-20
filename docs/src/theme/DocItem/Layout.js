import React, { useEffect } from 'react';
import Layout from '@theme-original/DocItem/Layout';

export default function DocItemLayout(props) {
  useEffect(() => {
    const toc = document.querySelector('.table-of-contents');
    if (toc && !toc.querySelector('.global-logo-corner')) {
      const wrapper = document.createElement('div');
      wrapper.className = 'global-logo-corner';

      const logo = document.createElement('img');
      logo.src = '/img/iciclelogo.png';
      logo.alt = 'ICICLE Logo';

      wrapper.appendChild(logo);
      toc.prepend(wrapper);
    }
  }, []);

  return <Layout {...props} />;
}
