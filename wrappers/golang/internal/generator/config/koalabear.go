package config

func init() {
	var koalabear = FieldData{
		PackageName:       "koalabear",
		Field:             "koalabear",
		LimbsNum:          1,
		SupportsExtension: true,
		ExtensionLimbsNum: 4,
		SupportsNTT:       true,
		SupportsPoseidon:  true,
		SupportsPoseidon2: true,
		ROU:               1791270792,
	}

	addField(koalabear)
}
