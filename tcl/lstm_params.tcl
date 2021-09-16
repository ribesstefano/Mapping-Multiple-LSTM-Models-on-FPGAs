proc append_lstm_params {&defines} {
    dict set params NUM_GATES 4
    dict set params NUM_INPUTS 2
    dict set params NUM_SAMPLES 2
    dict set params INPUT_SIZE 128
    dict set params HIDDEN_SIZE 64
    dict set params NUM_ITERATIONS 32
    dict set params NUM_TILES_U 8
    dict set params NUM_ZERO_TILES_U 2
    dict set params NUM_TILES_V 8
    dict set params NUM_ZERO_TILES_V 2
    dict set params NUM_TIMESTEPS 28
    dict set params FIX_WIDTH 16
    dict set params FIX_FRACT_WIDTH 6

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