// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2/native/extension/module.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fairseq2/native/data/data_processor.h>
#include <fairseq2/native/data/text/sentencepiece/sentencepiece.h>

namespace py = pybind11;

namespace fairseq2 {

void
def_sentencepiece(py::module_ &text_module)
{
    py::module_ m = text_module.def_submodule("sentencepiece");

    py::class_<sp_model, std::shared_ptr<sp_model>>(m, "SentencePieceModel")
        .def(
            py::init([](
                std::string_view pathname, std::optional<std::vector<std::string>> control_symbols)
            {
                sp_model_options opts{};

                if (control_symbols)
                    opts.control_symbols() = *std::move(control_symbols);

                return std::make_unique<sp_model>(pathname, std::move(opts));
            }),
            py::arg("pathname"),
            py::arg("control_symbols") = std::nullopt)

        .def(
            py::pickle(
                [](const sp_model &self)
                {
                    std::string serialized = self.serialize();

                    return py::bytes(serialized);
                },
                [](const py::bytes &bits)
                {
                    auto serialized = bits.cast<std::string>();

                    return sp_model::from_serialized(serialized);
                }))

        .def("token_to_index", &sp_model::token_to_index)
        .def("index_to_token", &sp_model::index_to_token)

        .def_property_readonly("unk_idx", &sp_model::unk_idx)
        .def_property_readonly("bos_idx", &sp_model::bos_idx)
        .def_property_readonly("eos_idx", &sp_model::eos_idx)
        .def_property_readonly("pad_idx", &sp_model::pad_idx)

        .def_property_readonly("vocab_size", &sp_model::vocab_size);

    py::class_<sp_encoder, data_processor, std::shared_ptr<sp_encoder>>(m, "SentencePieceEncoder")
        .def(
            py::init([](
                std::shared_ptr<const sp_model> model,
                std::optional<std::vector<std::string>> prefix_tokens,
                std::optional<std::vector<std::string>> suffix_tokens,
                bool reverse,
                bool enable_sampling,
                std::int32_t nbest_size,
                float alpha,
                std::optional<at::Device> device,
                bool pin_memory)
            {
                auto opts = sp_encoder_options()
                    .reverse(reverse)
                    .enable_sampling(enable_sampling)
                    .nbest_size(nbest_size)
                    .alpha(alpha)
                    .device(device)
                    .pin_memory(pin_memory);

                if (prefix_tokens)
                    opts.prefix_tokens() = *std::move(prefix_tokens);

                if (suffix_tokens)
                    opts.suffix_tokens() = *std::move(suffix_tokens);

                return sp_encoder{std::move(model), std::move(opts)};
            }),
            py::arg("model"),
            py::arg("prefix_tokens")   = std::nullopt,
            py::arg("suffix_tokens")   = std::nullopt,
            py::arg("reverse")         = false,
            py::arg("enable_sampling") = false,
            py::arg("nbest_size")      = -1,
            py::arg("alpha")           = 0.1,
            py::arg("device")          = std::nullopt,
            py::arg("pin_memory")      = false);

    py::class_<sp_decoder, data_processor, std::shared_ptr<sp_decoder>>(m, "SentencePieceDecoder")
        .def(
            py::init<std::shared_ptr<const sp_model>, bool>(),
            py::arg("model"),
            py::arg("reverse") = false);
}

}  // namespace fairseq2
