#
# @brief      Find all files in a directory and return them in a list.
#
# @param      basedir            The directory to start looking in pattern.
# @param      pattern            A pattern, as defined by the glob command, that
#                                the files must match.
# @param      exclude_dirs_list  Ignore searching in specified directories
#
# @return     The list of found files.
#
proc findFiles { basedir pattern exclude_dirs_list } {
    # Fix the directory name, this ensures the directory name is in the
    # native format for the platform and contains a final directory seperator
    set basedir [string trimright [file join [file normalize $basedir] { }]]
    set fileList {}
    # Look in the current directory for matching files, -type {f r}
    # means ony readable normal files are looked at, -nocomplain stops
    # an error being thrown if the returned list is empty
    foreach fileName [glob -nocomplain -type {f r} -path $basedir $pattern] {
        lappend fileList $fileName
    }
    # Now look for any sub direcories in the current directory
    foreach dirName [glob -nocomplain -type {d  r} -path $basedir *] {
        # Recusively call the routine on the sub directory and append any
        # new files to the results
        if {[lsearch -exact ${exclude_dirs_list} $dirName] == -1} {
            set subDirList [findFiles $dirName $pattern $exclude_dirs_list]
            if { [llength $subDirList] > 0 } {
                foreach subDirFile $subDirList {
                    lappend fileList $subDirFile
                }
            }
        }
    }
    return $fileList
}

#
# @brief      Greps a file content and writes matches to a file.
#
# @param      re     Regular expression
# @param      lines  Number of lines to report/include after the found match
# @param      fin    The fin pointer
# @param      fout   The fout pointer
#
proc grep {re lines fin fout} {
    set cnt 0
    set match false
    seek $fin 0
    while {[gets $fin line] >= 0} {
        if [regexp -- $re $line] {
            set cnt 0
            set match true
        }
        if {$match && ($cnt < $lines)} {
            puts $line
            puts $fout $line
            set cnt [expr {$cnt +1}]
        } else {
            set match false
        }
    }
}