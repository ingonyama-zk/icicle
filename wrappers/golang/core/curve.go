package core

type Projective struct {
	X, Y, Z Field
}

func (p* Projective) Zero() Projective {
	p.X.Zero()
	p.Y.Zero()
	p.Z.Zero()
	
	return *p
}

func (p* Projective) FromLimbs(x, y, z []uint32) Projective {
	p.X.FromLimbs(x)
	p.Y.FromLimbs(y)
	p.Z.FromLimbs(z)

	return *p
}

func (p* Projective) FromAffine(a Affine) Projective {
	z := &Field {
		Limbs: make([]uint32, len(a.X.Limbs)),
	}
	z.One()
	
	return Projective{
		X: a.X,
		Y: a.Y,
		Z: *z,
	}
}

type Affine struct {
	X, Y Field
}

func (a* Affine) Zero() Affine {
	a.X.Zero()
	a.Y.Zero()
	
	return *a
}

func (a* Affine) FromLimbs(x, y []uint32) Affine {
	a.X.FromLimbs(x)
	a.Y.FromLimbs(y)

	return *a
}

func (a* Affine) ToProjective() Projective {
	z := &Field {
		Limbs: make([]uint32, len(a.X.Limbs)),
	}
	z.One()
	
	return Projective{
		X: a.X,
		Y: a.Y,
		Z: *z,
	}
}
