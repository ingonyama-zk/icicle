.PHONY: updateicicle copy_folder

updateicicle: copy_folder
	@echo "Icicle updated successfully!"

copy_folder:
	mkdir -p goicicle/icicle/
	cp -r icicle/ goicicle/icicle/
