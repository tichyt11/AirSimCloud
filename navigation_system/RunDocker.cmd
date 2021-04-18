SET mypath=%~dp0

CALL docker run -ti -w /working -v %mypath%:/working omvgsnp