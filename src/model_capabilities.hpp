#pragma once

#include <qhexrt/qhexrt_c.h>

#include <string_view>

namespace qhx {

const qhx_model_capability* find_model_capability_by_manifest(
    std::string_view manifest_name) noexcept;

}  // namespace qhx
