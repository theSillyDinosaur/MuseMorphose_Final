from miditok import REMI, TokenizerConfig
from symusic import Score

BEAT_RES = {(0, 1): 24, (1, 2): 8, (2, 4): 4, (4, 8): 2}
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "num_velocities": 24,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,  # no multitrack here
    "num_tempos": 32,
    "tempo_range": (50, 200),  # (min_tempo, max_tempo)
}

# Creating a multitrack tokenizer, read the doc to explore all the parameters
config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

def remi2midi_classic(tokens, output_file):
  converted_back_midi = tokenizer.decode(tokens)
  print(converted_back_midi)
  converted_back_midi.dump_midi(output_file)