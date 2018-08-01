#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "imutil" for configuration "Release"
set_property(TARGET imutil APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(imutil PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/sift3d/libimutil.so"
  IMPORTED_SONAME_RELEASE "libimutil.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS imutil )
list(APPEND _IMPORT_CHECK_FILES_FOR_imutil "${_IMPORT_PREFIX}/lib/sift3d/libimutil.so" )

# Import target "sift3D" for configuration "Release"
set_property(TARGET sift3D APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sift3D PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/sift3d/libsift3D.so"
  IMPORTED_SONAME_RELEASE "libsift3D.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS sift3D )
list(APPEND _IMPORT_CHECK_FILES_FOR_sift3D "${_IMPORT_PREFIX}/lib/sift3d/libsift3D.so" )

# Import target "reg" for configuration "Release"
set_property(TARGET reg APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(reg PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/sift3d/libreg.so"
  IMPORTED_SONAME_RELEASE "libreg.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS reg )
list(APPEND _IMPORT_CHECK_FILES_FOR_reg "${_IMPORT_PREFIX}/lib/sift3d/libreg.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
