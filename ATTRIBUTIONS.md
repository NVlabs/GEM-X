This repository uses the following third-party components. Their respective licenses are reproduced below.

---

## OpenAI — guided-diffusion

https://github.com/openai/guided-diffusion

The following files are derived from OpenAI's guided-diffusion codebase (a PyTorch port of Ho et al., "Denoising Diffusion Probabilistic Models"):

- `gem/diffusion_utils/gaussian_diffusion.py`
- `gem/diffusion_utils/losses.py`
- `gem/diffusion_utils/nn.py`
- `gem/diffusion_utils/respace.py`

**License:** MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Meta Platforms / Facebook Research — PyTorch3D / ACTOR

https://github.com/facebookresearch/pytorch3d
https://github.com/Mathux/ACTOR

The following file is derived from Facebook's PyTorch3D library (via the ACTOR project):

- `gem/utils/rotation_conversions.py`

Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

**License:** BSD 3-Clause License

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## Megvii — YOLOX

https://github.com/Megvii-BaseDetection/YOLOX

The following file contains a self-contained YOLOX ONNX detector reimplementation:

- `gem/utils/yolox_detector.py`

Copyright (c) 2021 Megvii, Inc. All rights reserved.

**License:** Apache License 2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

## Megvii — ByteTrack

https://github.com/ifzhang/ByteTrack

The following file contains a self-contained ByteTrack multi-object tracker reimplementation:

- `gem/utils/yolox_detector.py`

Copyright (c) 2021 Yifu Zhang, licensed under the MIT License.

**License:** MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## SAM 3D Body

https://github.com/facebookresearch/sam-3d-body

SAM License
Last Updated: November 19, 2025

"Agreement" means the terms and conditions for use, reproduction, distribution and modification of the SAM Materials set forth herein.

"SAM Materials" means, collectively, Documentation and the models, software and algorithms, including machine-learning model code, trained model weights, inference-enabling code, training-enabling code, fine-tuning enabling code, and other elements of the foregoing distributed by Meta and made available under this Agreement.

"Documentation" means the specifications, manuals and documentation accompanying
SAM Materials distributed by Meta.

"Licensee" or "you" means you, or your employer or any other person or entity (if you are entering into this Agreement on such person or entity's behalf), of the age required under applicable laws, rules or regulations to provide legal consent and that has legal authority to bind your employer or such other person or entity if you are entering in this Agreement on their behalf.

"Meta" or "we" means Meta Platforms Ireland Limited (if you are located in or, if you are an entity, your principal place of business is in the EEA or Switzerland) or Meta Platforms, Inc. (if you are located outside of the EEA or Switzerland).

"Sanctions" means any economic or trade sanctions or restrictions administered or enforced by the United States (including the Office of Foreign Assets Control of the U.S. Department of the Treasury ("OFAC"), the U.S. Department of State and the U.S. Department of Commerce), the United Nations, the European Union, or the United Kingdom.

"Trade Controls" means any of the following: Sanctions and applicable export and import controls.

By using or distributing any portion or element of the SAM Materials, you agree to be bound by this Agreement.

1. License Rights and Redistribution.

a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license under Meta's intellectual property or other rights owned by Meta embodied in the SAM Materials to use, reproduce, distribute, copy, create derivative works of, and make modifications to the SAM Materials.

i. Grant of Patent License. Subject to the terms and conditions of this License, you are granted a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by Meta that are necessarily infringed alone or by combination of their contribution(s) with the SAM 3 Materials. If you institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the SAM 3 Materials incorporated within the work constitutes direct or contributory patent infringement, then any patent licenses granted to you under this License for that work shall terminate as of the date such litigation is filed.

b. Redistribution and Use.

i. Distribution of SAM Materials, and any derivative works thereof, are subject to the terms of this Agreement. If you distribute or make the SAM Materials, or any derivative works thereof, available to a third party, you may only do so under the terms of this Agreement and you shall provide a copy of this Agreement with any such SAM Materials.

ii. If you submit for publication the results of research you perform on, using, or otherwise in connection with SAM Materials, you must acknowledge the use of SAM Materials in your publication.

iii. Your use of the SAM Materials must comply with applicable laws and regulations, including Trade Control Laws and applicable privacy and data protection laws.

iv. Your use of the SAM Materials will not involve or encourage others to reverse engineer, decompile or discover the underlying components of the SAM Materials.

v. You are not the target of Trade Controls and your use of SAM Materials must comply with Trade Controls. You agree not to use, or permit others to use, SAM Materials for any activities subject to the International Traffic in Arms Regulations (ITAR) or end uses prohibited by Trade Controls, including those related to military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons.

2. User Support. Your use of the SAM Materials is done at your own discretion; Meta does not process any information nor provide any service in relation to such use. Meta is under no obligation to provide any support services for the SAM Materials. Any support provided is "as is", "with all faults", and without warranty of any kind.

3. Disclaimer of Warranty. UNLESS REQUIRED BY APPLICABLE LAW, THE SAM MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING THE SAM MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE SAM MATERIALS AND ANY OUTPUT AND RESULTS.

4. Limitation of Liability. IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT, FOR ANY LOST PROFITS OR ANY DIRECT OR INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.

5. Intellectual Property.

a. Subject to Meta's ownership of SAM Materials and derivatives made by or for Meta, with respect to any derivative works and modifications of the SAM Materials that are made by you, as between you and Meta, you are and will be the owner of such derivative works and modifications.

b. If you institute litigation or other proceedings against Meta or any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the SAM Materials, outputs or results, or any portion of any of the foregoing, constitutes infringement of intellectual property or other rights owned or licensable by you, then any licenses granted to you under this Agreement shall terminate as of the date such litigation or claim is filed or instituted. You will indemnify and hold harmless Meta from and against any claim by any third party arising out of or related to your use or distribution of the SAM Materials.

6. Term and Termination. The term of this Agreement will commence upon your acceptance of this Agreement or access to the SAM Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein. Meta may terminate this Agreement if you are in breach of any term or condition of this Agreement. Upon termination of this Agreement, you shall delete and cease use of the SAM Materials. Sections 3, 4 and 7 shall survive the termination of this Agreement.

7. Governing Law and Jurisdiction. This Agreement will be governed and construed under the laws of the State of California without regard to choice of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The courts of California shall have exclusive jurisdiction of any dispute arising out of this Agreement.

8. Modifications and Amendments. Meta may modify this Agreement from time to time; provided that they are similar in spirit to the current version of the Agreement, but may differ in detail to address new problems or concerns. All such changes will be effective immediately. Your continued use of the SAM Materials after any modification to this Agreement constitutes your agreement to such modification. Except as provided in this Agreement, no modification or addition to any provision of this Agreement will be binding unless it is in writing and signed by an authorized representative of both you and Meta.
