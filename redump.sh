dumpdir=""
state_dir=""

for name in train dev; do
    echo "Dump ${name} set to ark files ${dumpdir}/${name}/arks/wav.*.ark"
    # torchrun --nproc_per_node=32 --master_port=1234 scripts/dump_to_wav_ark.py \
    #   --wav_scp ${dumpdir}/${name}_24k/wav.scp \
    #   --out_dir ${dumpdir}/${name}/arks \
    #   --sample_rate 16000

    # mkdir -p ${dumpdir}/${name} exp/${state_dir}/${name}
    # cat ${dumpdir}/${name}/arks/wav.*.scp | sort > ${dumpdir}/${name}/wav.scp
    cat ${dumpdir}/${name}/arks/length.*.txt | shuf | awk '{print $1,int($2/640)}' > exp/${state_dir}/${name}/codec_shape

    # echo "Collect and tokenize text files of ${name} into one phoneme file"
    # python scripts/collect_text_flist_to_phone_scp.py \
    #   ${dumpdir}/${name}_24k/normalized_txt.flist \
    #   ${dumpdir}/${name}/phoneme
  done