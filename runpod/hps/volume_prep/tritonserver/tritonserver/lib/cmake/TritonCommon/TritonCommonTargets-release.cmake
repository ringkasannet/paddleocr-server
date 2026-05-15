#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TritonCommon::triton-common-async-work-queue" for configuration "Release"
set_property(TARGET TritonCommon::triton-common-async-work-queue APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TritonCommon::triton-common-async-work-queue PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtritonasyncworkqueue.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS TritonCommon::triton-common-async-work-queue )
list(APPEND _IMPORT_CHECK_FILES_FOR_TritonCommon::triton-common-async-work-queue "${_IMPORT_PREFIX}/lib/libtritonasyncworkqueue.a" )

# Import target "TritonCommon::triton-common-error" for configuration "Release"
set_property(TARGET TritonCommon::triton-common-error APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TritonCommon::triton-common-error PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtritoncommonerror.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS TritonCommon::triton-common-error )
list(APPEND _IMPORT_CHECK_FILES_FOR_TritonCommon::triton-common-error "${_IMPORT_PREFIX}/lib/libtritoncommonerror.a" )

# Import target "TritonCommon::triton-common-table-printer" for configuration "Release"
set_property(TARGET TritonCommon::triton-common-table-printer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TritonCommon::triton-common-table-printer PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtritontableprinter.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS TritonCommon::triton-common-table-printer )
list(APPEND _IMPORT_CHECK_FILES_FOR_TritonCommon::triton-common-table-printer "${_IMPORT_PREFIX}/lib/libtritontableprinter.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
