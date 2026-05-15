#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TritonBackend::triton-backend-utils" for configuration "Release"
set_property(TARGET TritonBackend::triton-backend-utils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TritonBackend::triton-backend-utils PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtritonbackendutils.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS TritonBackend::triton-backend-utils )
list(APPEND _IMPORT_CHECK_FILES_FOR_TritonBackend::triton-backend-utils "${_IMPORT_PREFIX}/lib/libtritonbackendutils.a" )

# Import target "TritonBackend::kernel-library-new" for configuration "Release"
set_property(TARGET TritonBackend::kernel-library-new APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TritonBackend::kernel-library-new PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libkernel-library-new.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS TritonBackend::kernel-library-new )
list(APPEND _IMPORT_CHECK_FILES_FOR_TritonBackend::kernel-library-new "${_IMPORT_PREFIX}/lib/libkernel-library-new.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
