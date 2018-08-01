# Install script for directory: /mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab" TYPE FILE FILES
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/setupSift3D.m"
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/imRead3D.m"
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/imWrite3D.m"
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/detectSift3D.m"
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/extractSift3D.m"
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/orientation3D.m"
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/keypoint3D.m"
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/registerSift3D.m"
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/matchSift3D.m"
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/checkUnits3D.m"
    "/mp/nas1/fixstars/karl/SIFT3D/wrappers/matlab/Sift3DParser.m"
    )
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/libmexutil.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/libmexutil.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/libmexutil.so"
         RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab" TYPE SHARED_LIBRARY FILES "/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab/libmexutil.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/libmexutil.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/libmexutil.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/libmexutil.so"
         OLD_RPATH "/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64::::"
         NEW_RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/libmexutil.so")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImRead3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImRead3D.mexa64")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImRead3D.mexa64"
         RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab" TYPE SHARED_LIBRARY FILES "/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab/mexImRead3D.mexa64")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImRead3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImRead3D.mexa64")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImRead3D.mexa64"
         OLD_RPATH "/usr/local/MATLAB/R2018a/bin/glnxa64:/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab::::"
         NEW_RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImRead3D.mexa64")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImWrite3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImWrite3D.mexa64")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImWrite3D.mexa64"
         RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab" TYPE SHARED_LIBRARY FILES "/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab/mexImWrite3D.mexa64")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImWrite3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImWrite3D.mexa64")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImWrite3D.mexa64"
         OLD_RPATH "/usr/local/MATLAB/R2018a/bin/glnxa64:/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab::::"
         NEW_RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexImWrite3D.mexa64")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexDetectSift3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexDetectSift3D.mexa64")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexDetectSift3D.mexa64"
         RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab" TYPE SHARED_LIBRARY FILES "/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab/mexDetectSift3D.mexa64")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexDetectSift3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexDetectSift3D.mexa64")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexDetectSift3D.mexa64"
         OLD_RPATH "/usr/local/MATLAB/R2018a/bin/glnxa64:/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab::::"
         NEW_RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexDetectSift3D.mexa64")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexOrientation3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexOrientation3D.mexa64")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexOrientation3D.mexa64"
         RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab" TYPE SHARED_LIBRARY FILES "/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab/mexOrientation3D.mexa64")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexOrientation3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexOrientation3D.mexa64")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexOrientation3D.mexa64"
         OLD_RPATH "/usr/local/MATLAB/R2018a/bin/glnxa64:/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab::::"
         NEW_RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexOrientation3D.mexa64")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexExtractSift3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexExtractSift3D.mexa64")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexExtractSift3D.mexa64"
         RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab" TYPE SHARED_LIBRARY FILES "/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab/mexExtractSift3D.mexa64")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexExtractSift3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexExtractSift3D.mexa64")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexExtractSift3D.mexa64"
         OLD_RPATH "/usr/local/MATLAB/R2018a/bin/glnxa64:/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab::::"
         NEW_RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexExtractSift3D.mexa64")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexRegisterSift3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexRegisterSift3D.mexa64")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexRegisterSift3D.mexa64"
         RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab" TYPE SHARED_LIBRARY FILES "/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab/mexRegisterSift3D.mexa64")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexRegisterSift3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexRegisterSift3D.mexa64")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexRegisterSift3D.mexa64"
         OLD_RPATH "/usr/local/MATLAB/R2018a/bin/glnxa64:/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab::::"
         NEW_RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexRegisterSift3D.mexa64")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexMatchSift3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexMatchSift3D.mexa64")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexMatchSift3D.mexa64"
         RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab" TYPE SHARED_LIBRARY FILES "/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab/mexMatchSift3D.mexa64")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexMatchSift3D.mexa64" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexMatchSift3D.mexa64")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexMatchSift3D.mexa64"
         OLD_RPATH "/usr/local/MATLAB/R2018a/bin/glnxa64:/mp/nas1/fixstars/karl/SIFT3D/build/lib/wrappers/matlab::::"
         NEW_RPATH "/usr/local/lib/sift3d:/usr/local/lib/sift3d/wrappers/matlab:/usr/local/MATLAB/R2018a/bin/glnxa64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/sift3d/wrappers/matlab/mexMatchSift3D.mexa64")
    endif()
  endif()
endif()

