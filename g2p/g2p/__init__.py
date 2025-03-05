# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from g2p.g2p import cleaners
from tokenizers import Tokenizer
from g2p.g2p.text_tokenizers import TextTokenizer
import LangSegment
import json
import re


class PhonemeBpeTokenizer:

    def __init__(self, vacab_path="./g2p/g2p/vocab.json"):
        self.lang2backend = {
            "zh": "zh",  # Changed from "cmn" to "zh"
            "ja": "ja",
            "en": "en-us",
            "fr": "fr-fr",
            "ko": "ko",
            "de": "de",
        }
        self.text_tokenizers = {}
        self.int_text_tokenizers()

        with open(vacab_path, "r") as f:
            json_data = f.read()
        data = json.loads(json_data)
        self.vocab = data["vocab"]
        LangSegment.setfilters(["en", "zh", "ja", "ko", "fr", "de"])

    def int_text_tokenizers(self):
        # Initialize English first as our fallback
        try:
            self.text_tokenizers["en"] = TextTokenizer(language="en-us")
        except Exception as e:
            print(f"Error initializing English tokenizer: {e}")
            raise RuntimeError("Cannot initialize the required English tokenizer")
            
        # Initialize other languages, falling back to English if they fail
        for key, value in self.lang2backend.items():
            if key == "en":
                continue  # Already initialized
            try:
                self.text_tokenizers[key] = TextTokenizer(language=value)
            except Exception as e:
                print(f"Warning: Could not initialize tokenizer for language {key} ({value}): {e}")
                print(f"Using English tokenizer as fallback for {key}")
                self.text_tokenizers[key] = self.text_tokenizers["en"]

    def tokenize(self, text, sentence, language):
        # Always default to English if language is not supported
        if language not in self.text_tokenizers:
            print(f"Language {language} not supported, falling back to English")
            language = "en"

        # 1. convert text to phoneme
        phonemes = []
        if language == "auto":
            # Limit language detection to supported languages
            supported_langs = list(self.text_tokenizers.keys())
            LangSegment.setfilters(supported_langs)
            seglist = LangSegment.getTexts(text)
            tmp_ph = []
            for seg in seglist:
                # Make sure we use a supported language
                seg_lang = seg["lang"] if seg["lang"] in self.text_tokenizers else "en"
                tmp_ph.append(
                    self._clean_text(
                        seg["text"], sentence, seg_lang, ["cjekfd_cleaners"]
                    )
                )
            phonemes = "|_|".join(tmp_ph)
        else:
            phonemes = self._clean_text(text, sentence, language, ["cjekfd_cleaners"])
        # print('clean text: ', phonemes)

        # 2. tokenize phonemes
        phoneme_tokens = self.phoneme2token(phonemes)
        # print('encode: ', phoneme_tokens)

        # # 3. decode tokens [optional]
        # decoded_text = self.tokenizer.decode(phoneme_tokens)
        # print('decoded: ', decoded_text)

        return phonemes, phoneme_tokens

    def _clean_text(self, text, sentence, language, cleaner_names):
        for name in cleaner_names:
            cleaner = getattr(cleaners, name)
            if not cleaner:
                raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text, sentence, language, self.text_tokenizers)
        return text

    def phoneme2token(self, phonemes):
        tokens = []
        if isinstance(phonemes, list):
            for phone in phonemes:
                phone = phone.split("\t")[0]
                phonemes_split = phone.split("|")
                tokens.append(
                    [self.vocab[p] for p in phonemes_split if p in self.vocab]
                )
        else:
            phonemes = phonemes.split("\t")[0]
            phonemes_split = phonemes.split("|")
            tokens = [self.vocab[p] for p in phonemes_split if p in self.vocab]
        return tokens
