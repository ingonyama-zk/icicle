package internal


type MockProjective struct {
	X, Y, Z MockField
}

func (p MockProjective) Size() int {
	return p.X.Size() * 3
}

func (p MockProjective) AsPointer() *uint64 {
	return p.X.AsPointer()
}

func (p *MockProjective) Zero() MockProjective {
	p.X.Zero()
	p.Y.One()
	p.Z.Zero()

	return *p
}

func (p *MockProjective) FromLimbs(x, y, z []uint64) MockProjective {
	p.X.FromLimbs(x)
	p.Y.FromLimbs(y)
	p.Z.FromLimbs(z)

	return *p
}

func (p *MockProjective) FromAffine(a MockAffine) MockProjective {
	z := MockField{}
	z.One()

	p.X = a.X
	p.Y = a.Y
	p.Z = z

	return *p
}

type MockAffine struct {
	X, Y MockField
}

func (a MockAffine) Size() int {
	return a.X.Size() * 2
}

func (a MockAffine) AsPointer() *uint64 {
	return a.X.AsPointer()
}

func (a *MockAffine) Zero() MockAffine {
	a.X.Zero()
	a.Y.Zero()

	return *a
}

func (a *MockAffine) FromLimbs(x, y []uint64) MockAffine {
	a.X.FromLimbs(x)
	a.Y.FromLimbs(y)

	return *a
}

func (a MockAffine) ToProjective() MockProjective {
	var z MockField

	return MockProjective{
		X: a.X,
		Y: a.Y,
		Z: z.One(),
	}
}
