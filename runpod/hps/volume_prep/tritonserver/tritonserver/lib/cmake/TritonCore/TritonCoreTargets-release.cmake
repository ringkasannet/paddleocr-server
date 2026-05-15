#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TritonCore::triton-core-serverstub" for configuration "Release"
set_property(TARGET TritonCore::triton-core-serverstub APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TritonCore::triton-core-serverstub PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtritonserver_stub.so"
  IMPORTED_NO_SONAME_RELEASE "TRUE"
  )

list(APPEND _IMPORT_CHECK_TARGETS TritonCore::triton-core-serverstub )
list(APPEND _IMPORT_CHECK_FILES_FOR_TritonCore::triton-core-serverstub "${_IMPORT_PREFIX}/lib/libtritonserver_stub.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
