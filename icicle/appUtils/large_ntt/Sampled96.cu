/*

Copyright (c) 2023 Yrrid Software, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the �Software�), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED �AS IS�, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

__device__ __forceinline__ uint32_t sign(uint32_t x) {
  return ((int32_t)x)>>31;
}

__device__ __forceinline__ uint64_t mul(uint64_t a, uint64_t b) {
  uint64_t odd, even0, even1;
  uint32_t l, m, h, hh, madd;

  even0=mulwide(ulow(a), ulow(b));
  even1=mulwide(uhigh(a), uhigh(b));

  odd=mulwide(ulow(a), uhigh(b));
  odd=madwide_cc(uhigh(a), ulow(b), odd);
  hh=addc(0, 0);

  l=ulow(even0);
  m=add_cc(uhigh(even0), ulow(odd));
  h=addc_cc(ulow(even1), uhigh(odd));
  hh=addc(uhigh(even1), hh);

  l=sub_cc(l, h);
  madd=subc(h, 0);
  m=add_cc(m, madd);
  h=addc(0, 0);

  l=sub_cc(l, hh);
  m=subc_cc(m, 0);
  h=subc(h, 0);

  l=add_cc(l, -h);
  m=addc(m, sign(h));
  return make_wide(l, m);
}

__device__ __forceinline__ uint64_t root1024(uint32_t fiveBits) {
  uint64_t root1=1, root2=1, root3=1;

  // compute power(root1k, threadIdx.x & 0x3F)
  if((fiveBits & 0x03) == 0x01)
    root1=0x870C4F2B1519AAD6ull;
  else if((fiveBits & 0x03) == 0x02)
    root1=0x9CA35B58AFC843F2ull;
  else if((fiveBits & 0x03) == 0x03)
    root1=0x90A39AFA462328B2ull;

  if((fiveBits & 0x0C) == 0x04)
    root2=0xB2A043752E8CF9ACull;
  else if((fiveBits & 0x0C) == 0x08)
    root2=0xFFFFFDFF02000201ull;
  else if((fiveBits & 0x0C) == 0x0C)
    root2=0xD1CF03DEB11501D7ull;

  if((fiveBits & 0x10) == 0x10)
    root3=0x0000000000000008ull;

  return mul(root1, mul(root2, root3));
}
  
__device__ __forceinline__ uint64_t root4096(uint32_t sixBits) {
  uint64_t root1=1, root2=1, root3=1;

  // compute power(root4k, threadIdx.x & 0x3F)

  if((sixBits & 0x03) == 0x01)
    root1=0xD97FD464F2A78E28ull;
  else if((sixBits & 0x03) == 0x02)
    root1=0x67CAB0A01E964566ull;
  else if((sixBits & 0x03) == 0x03)
    root1=0xA3F7AE885A25197Bull;

  if((sixBits & 0x0C) == 0x04)
    root2=0x870C4F2B1519AAD6ull;
  else if((sixBits & 0x0C) == 0x08)
    root2=0x9CA35B58AFC843F2ull;
  else if((sixBits & 0x0C) == 0x0C)
    root2=0x90A39AFA462328B2ull;

  if((sixBits & 0x30) == 0x10)
    root3=0xB2A043752E8CF9ACull;
  else if((sixBits & 0x30) == 0x20)
    root3=0xFFFFFDFF02000201ull;
  else if((sixBits & 0x30) == 0x30)
    root3=0xD1CF03DEB11501D7ull;

  return mul(root1, mul(root2, root3));
}

class Sampled96 {
  public:
  
  typedef int4 Value;
  
  static __device__ __forceinline__ Value add(const Value& a, const Value& b) {
    Value r;
    
    // input and output are sampled values
    
    r.x=a.x + b.x;
    r.y=a.y + b.y;
    r.z=a.z + b.z;
    r.w=a.w + b.w;
    return r;
  }
  
  static __device__ __forceinline__ Value sub(const Value& a, const Value& b) {
    Value r;
    
    // input and output are sampled values
    
    r.x=a.x - b.x;
    r.y=a.y - b.y;
    r.z=a.z - b.z;
    r.w=a.w - b.w;
    return r;
  }

  template<bool condensed>
  static __device__ __forceinline__ Value mul(uint64_t a, uint64_t b) {
    uint64_t odd, even0, even1;
    uint32_t l, m, h, hh;
    Value    r;

    // inputs are normalized values
    // outputs a condensed value

    even0=mulwide(ulow(a), ulow(b));
    even1=mulwide(uhigh(a), uhigh(b));

    odd=mulwide(ulow(a), uhigh(b));
    odd=madwide_cc(uhigh(a), ulow(b), odd);
    hh=addc(0, 0);

    l=ulow(even0);
    m=add_cc(uhigh(even0), ulow(odd));
    h=addc_cc(ulow(even1), uhigh(odd));
    hh=addc(uhigh(even1), hh);

    if(condensed) {
      r.x=sub_cc(l, h);
      r.y=subc(h, 0);
    
      r.y=add_cc(r.y, m);
      r.z=addc(0, 0);
    
      r.x=sub_cc(r.x, hh);
      r.y=subc_cc(r.y, 0);
      r.z=subc(r.z, 0);
    }
    else {
      l=sub_cc(l, hh);
      m=subc_cc(m, 0);
      h=subc_cc(h, 0);
      hh=subc(0, 0);
      
      r.x=prmt(l, 0, 0x4210);
      r.y=prmt(l, m, 0x6543) & 0x00FFFFFF;
      r.z=prmt(m, h, 0x5432) & 0x00FFFFFF;
      r.w=prmt(h, hh, 0x4321);
    }
    return r;
  }

  template<uint32_t bytePosition> 
  static __device__ __forceinline__ Value getValue(uint64_t sample, uint32_t highBytes) {
    Value r;
    
    // output sampled value
    
    r.x=prmt(ulow(sample), 0, 0x4210);
    r.y=prmt(ulow(sample), uhigh(sample), 0x6543) & 0x00FFFFFF;
    if(bytePosition==0) 
      r.z=prmt(uhigh(sample), highBytes, 0xC432);
    else if(bytePosition==1) 
      r.z=prmt(uhigh(sample), highBytes, 0xD532);
    else if(bytePosition==2) 
      r.z=prmt(uhigh(sample), highBytes, 0xE632);
    else if(bytePosition==3) 
      r.z=prmt(uhigh(sample), highBytes, 0xF732);
    else
      r.z=0;
    r.w=0;
    return r;
  }
  
  template<uint32_t bytePosition> 
  static __device__ __forceinline__ void setValue(uint64_t& sample, uint32_t& highBytes, const Value& a) {
    // input condensed value
    
    sample=make_wide((uint32_t)a.x, (uint32_t)a.y);
    if(bytePosition==0)
      highBytes=prmt(highBytes, (uint32_t)a.z, 0x3214);
    else if(bytePosition==1)
      highBytes=prmt(highBytes, (uint32_t)a.z, 0x3240);
    else if(bytePosition==2)
      highBytes=prmt(highBytes, (uint32_t)a.z, 0x3410);
    else if(bytePosition==3)
      highBytes=prmt(highBytes, (uint32_t)a.z, 0x4210);
  }

  static __device__ __forceinline__ Value getNormalizedValue(uint64_t sample) {
    Value r;
    
    // output sampled value
    
    r.x=prmt(ulow(sample), 0, 0x4210);
    r.y=prmt(ulow(sample), uhigh(sample), 0x6543) & 0x00FFFFFF;
    r.z=prmt(uhigh(sample), 0, 0x4432);
    r.w=0;
    return r;
  }
  
  static __device__ __forceinline__ void setNormalizedValue(uint64_t& sample, const Value& a) {
    // input a normalized value
  
    sample=make_wide((uint32_t)a.x, (uint32_t)a.y);
  }

  template<uint32_t amount>
  static __device__ __forceinline__ Value shiftRoot8(const Value& a) {
    // input and output are sampled values
    if(amount==0)
      return a;
    else if(amount==1) 
      return make_int4(-a.w, a.x, a.y, a.z);
    else if(amount==2)
      return make_int4(-a.z, -a.w, a.x, a.y);
    else if(amount==3)
      return make_int4(-a.y, -a.z, -a.w, a.x);
    else if(amount==4)
      return make_int4(-a.x, -a.y, -a.z, -a.w);
    else if(amount==5)
      return make_int4(a.w, -a.x, -a.y, -a.z);
    else if(amount==6)
      return make_int4(a.z, a.w, -a.x, -a.y);
    else if(amount==7)
      return make_int4(a.y, a.z, a.w, -a.x);
    else
      return a;
  }    
  
  template<uint32_t amount>
  static __device__ __forceinline__ Value shiftRoot32(const Value& a) {
    const uint32_t left=(amount%4)*6, right=24-left;
    Value rotated;
    Value r;
    
    rotated=shiftRoot8<amount/4>(a);
    if(left==0)
      return rotated;
      
    r.x=rotated.x<<left;
    r.y=rotated.y<<left;
    r.z=rotated.z<<left;
    r.w=rotated.w<<left;
    
    r.x=(r.x & 0x00FFFFFF) - (rotated.w>>right);
    r.y=(r.y & 0x00FFFFFF) + (rotated.x>>right);
    r.z=(r.z & 0x00FFFFFF) + (rotated.y>>right);
    r.w=(r.w & 0x00FFFFFF) + (rotated.z>>right);
    return r;
  }
  
  static __device__ __forceinline__ Value normalize(const Value& a) {
    Value    resolved;
    Value    r;
    uint32_t l, m, h, hh, s;

    // input a sampled value
    // output a normalized value

    resolved.x=a.x - (a.w>>24);
    resolved.y=a.y + (resolved.x>>24);
    resolved.z=a.z + (resolved.y>>24);
    resolved.w=(a.w & 0x00FFFFFF) + (resolved.z>>24);

    l=prmt(resolved.x, resolved.y, 0x4210);
    m=prmt(resolved.y, resolved.z, 0x5421);
    h=prmt(resolved.z, resolved.w, 0x6542);
    hh=resolved.w>>24;
    s=sign(resolved.w);

    r.x=sub_cc(l, h);
    r.y=subc(h, 0);

    r.y=add_cc(r.y, m);
    r.z=addc(0, 0);

    r.x=sub_cc(r.x, hh);
    r.y=subc_cc(r.y, s);
    r.z=subc(r.z, s);

    s=((r.z | ~r.y)==0 && r.x!=0) ? 1 : r.z;

    r.x=add_cc(r.x, -s);
    r.y=addc(r.y, sign(s));
    return r;
  }

  static __device__ __forceinline__ void dump(const Value& value) {
    Value temp;
    
    temp=normalize(value);
    printf("%08X%08X\n", temp.y, temp.x);
  }
};
