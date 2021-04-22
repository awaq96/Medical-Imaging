ReadMe

ReadMe:
To run the programs, navigate to the interface directory from the command line.

To run the Image Noise Driver:
	Arguments:
		-i : image file path
		-m: mask type, the mask function name
		-c : cutoff value
		-o: order value
		-t : thickness
		-matrix: matrix filepath

	Ex: To filter brain.png with a ideal Lowpass filter at cutoff 100 use 
		“python3 ImageNoiseDriver.py -i images/brain.png -m idealLowpassFilter -c 100”

To run the Image Acquisition Driver:

	Arguments:
	 	-i : image file path
		-m: mask type, the mask function name
		-p : percentage value
		-r: radius
		-maj: major axis value
		-min: minor axis value
		-a: angle
		-w: width
		-l: length
		-rc: raycount
		-s: sparsity

	Ex: To filter cardiac.jpg with a cartesian patter at 30% use
		“python3 ImageAcquisitionDriver.py -i images/cardiac.jpg -m cartesianPattern -p .3”
