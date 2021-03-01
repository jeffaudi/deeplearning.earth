pub_folder = public

run:
	hugo server --watch --verbose=true

build:
	rm -rf $(pub_folder)
	hugo -d $(pub_folder)

