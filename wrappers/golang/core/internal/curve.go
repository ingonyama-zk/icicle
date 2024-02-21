package internal

type Projective struct {
	X, Y, Z MockField
}

func (p Projective) Size() int {
	return p.X.Size() * 3
}

func (p Projective) AsPointer() *uint32 {
	return p.X.AsPointer()
}

func (p *Projective) Zero() Projective {
	p.X.Zero()
	p.Y.Zero()
	p.Z.Zero()

	return *p
}

func (p *Projective) FromLimbs(x, y, z []uint32) Projective {
	p.X.FromLimbs(x)
	p.Y.FromLimbs(y)
	p.Z.FromLimbs(z)

	return *p
}

func (p *Projective) FromAffine(a Affine) Projective {
	var z MockField
	z.One()

	p.X = a.X
	p.Y = a.Y
	p.Z = z

	return *p
}

type Affine struct {
	X, Y MockField
}

func (a Affine) Size() int {
	return a.X.Size() * 2
}

func (a Affine) AsPointer() *uint32 {
	return a.X.AsPointer()
}

func (a *Affine) Zero() Affine {
	a.X.Zero()
	a.Y.Zero()

	return *a
}

func (a *Affine) FromLimbs(x, y []uint32) Affine {
	a.X.FromLimbs(x)
	a.Y.FromLimbs(y)

	return *a
}

func (a Affine) ToProjective() Projective {
	var z MockField
	z.One()

	return Projective{
		X: a.X,
		Y: a.Y,
		Z: z,
	}
}
