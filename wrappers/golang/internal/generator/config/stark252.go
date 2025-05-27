package config

func init() {
	var stark252 = FieldData{
		PackageName:       "stark252",
		Field:             "stark252",
		LimbsNum:          8,
		SupportsExtension: false,
		ExtensionLimbsNum: 0,
		SupportsNTT:       true,
		SupportsPoseidon:  true,
		SupportsPoseidon2: true,
	}

	addField(stark252)
}
