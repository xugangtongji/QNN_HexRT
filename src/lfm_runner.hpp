#pragma once

#include <qhexrt/qhexrt_c.h>

#include <string>

struct qhx_session;
struct qhx_model;

namespace qhx {

bool prepare_lfm_model(qhx_model& model, std::string& error);
void reset_lfm_session(qhx_session& session) noexcept;

qhx_status run_lfm(qhx_session& session, const qhx_inputs& inputs,
                   const qhx_gen_cfg& config, const qhx_generate_options* options,
                   qhx_token_cb callback, void* callback_user, qhx_output& output,
                   std::string& error);

}  // namespace qhx
