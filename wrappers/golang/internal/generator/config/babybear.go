package config

var babybear = FieldData {
	PackageName: "babybear",
	Field: "babybear",
	LimbsNum: 1,
	SupportsExtension: true,
	ExtensionLimbsNum: 4,
	SupportsNTT: true,
}

func init() {
	addField(babybear)
}
