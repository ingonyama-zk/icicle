package config

func init() {
	var babybear = FieldData{
		PackageName:       "babybear",
		Field:             "babybear",
		LimbsNum:          1,
		SupportsExtension: true,
		ExtensionLimbsNum: 4,
		SupportsNTT:       true,
		SupportsPoseidon:  true,
		SupportsPoseidon2: true,
	}

	addField(babybear)
}
