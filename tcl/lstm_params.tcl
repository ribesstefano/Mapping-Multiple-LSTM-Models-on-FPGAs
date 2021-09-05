proc append_lstm_params {&defines} {
    dict set params NUM_GATES 4
    dict set params NUM_INPUTS 2
    dict set params NUM_SAMPLES 2
    dict set params INPUT_SIZE 1024
    dict set params HIDDEN_SIZE 512
    dict set params NUM_ITERATIONS 32
    dict set params NUM_TILES_U 4
    dict set params NUM_ZERO_TILES_U 1
    dict set params NUM_TILES_V 4
    dict set params NUM_ZERO_TILES_V 1
    dict set params NUM_TIMESTEPS 28

	set tmp {}
	append tmp " "
    foreach key [dict keys $params] {
        set value [dict get $params $key]
        append tmp "-D${key}=${value} "
    }
    puts "================================================================"
    puts "\[INFO\] LSTM parameters:"
    puts $tmp
    puts "================================================================"
    upvar 1 ${&defines} defines ;# To have a "pass by reference" argument.
    append defines $tmp
}