package config

func init() {
	var babybear = FieldData{
		PackageName:       "babybear",
		Field:             "babybear",
		LimbsNum:          1,
		SupportsExtension: true,
		ExtensionLimbsNum: 4,
		SupportsNTT:       true,
		ROU:               1461624142,
	}

	addField(babybear)
}
