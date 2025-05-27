package config

func init() {
	var m31 = FieldData{
		PackageName:       "m31",
		Field:             "m31",
		LimbsNum:          1,
		SupportsExtension: true,
		ExtensionLimbsNum: 4,
		SupportsNTT:       false,
		SupportsPoseidon:  true,
		SupportsPoseidon2: true,
	}

	addField(m31)
}
