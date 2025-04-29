package config

func init() {
	var goldilocks = FieldData{
		PackageName:       "goldilocks",
		Field:             "goldilocks",
		LimbsNum:          2,
		SupportsExtension: false,
		ExtensionLimbsNum: 0,
		SupportsNTT:       true,
		SupportsPoseidon:  false,
		SupportsPoseidon2: true,
	}

	addField(goldilocks)
}
