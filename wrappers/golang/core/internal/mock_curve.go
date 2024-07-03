package internal

type MockProjective struct {
	X, Y, Z MockBaseField
}

func (p MockProjective) Size() int {
	return p.X.Size() * 3
}

func (p MockProjective) AsPointer() *uint32 {
	return p.X.AsPointer()
}

func (p *MockProjective) Zero() MockProjective {
	p.X.Zero()
	p.Y.One()
	p.Z.Zero()

	return *p
}

func (p *MockProjective) FromLimbs(x, y, z []uint32) MockProjective {
	p.X.FromLimbs(x)
	p.Y.FromLimbs(y)
	p.Z.FromLimbs(z)

	return *p
}

func (p *MockProjective) FromAffine(a MockAffine) MockProjective {
	z := MockBaseField{}
	z.One()

	if (a.X == z.Zero()) && (a.Y == z.Zero()) {
		p.X = z.Zero()
		p.Y = z.One()
		p.Z = z.Zero()
	} else {
		p.X = a.X
		p.Y = a.Y
		p.Z = z.One()
	}

	return *p
}

type MockAffine struct {
	X, Y MockBaseField
}

func (a MockAffine) Size() int {
	return a.X.Size() * 2
}

func (a MockAffine) AsPointer() *uint32 {
	return a.X.AsPointer()
}

func (a *MockAffine) Zero() MockAffine {
	a.X.Zero()
	a.Y.Zero()

	return *a
}

func (a *MockAffine) FromLimbs(x, y []uint32) MockAffine {
	a.X.FromLimbs(x)
	a.Y.FromLimbs(y)

	return *a
}

func (a MockAffine) ToProjective() MockProjective {
	var z MockBaseField

	if (a.X == z.Zero()) && (a.Y == z.Zero()) {
		return MockProjective{
			X: z.Zero(),
			Y: z.One(),
			Z: z.Zero(),
		}
	}

	return MockProjective{
		X: a.X,
		Y: a.Y,
		Z: z.One(),
	}
}
