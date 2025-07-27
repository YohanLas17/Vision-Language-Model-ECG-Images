

from transformers import AutoTokenizer

class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(self, processor, max_length, is_train):
        self.processor = processor
        self.max_length = max_length
        self.is_train = is_train
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, data):
        is_train = data[0].get("is_train", True)
        texts, texts2, images = [], [], []

        # ------------------------------------------------------------------
        # Loop through each sample in the worker-batch
        # ------------------------------------------------------------------
        for item in data:
            # Safety net: skip empty or malformed samples
            if not item or "question" not in item or "answer" not in item:
                continue

            # Build chat templates
            texts.append(
                self.processor.apply_chat_template(
                    self.build_message(item["question"], item["answer"]),
                    add_generation_prompt=False,
                ).strip()
            )
            texts2.append(
                self.processor.apply_chat_template(
                    self.build_message(item["question"], None),
                    add_generation_prompt=True,
                ).strip()
            )
            # Expect a single PIL image or tensor here
            images.append([item["images"]])

        # If *all* items were skipped, raise an explicit error so you notice
        if len(texts) == 0:
            raise RuntimeError("Every sample in this worker batch was skipped (empty or malformed).")

        # ------------------------------------------------------------------
        # Build model inputs
        # ------------------------------------------------------------------
        if is_train:
            batch_full = self.processor(text=texts,  images=images, return_tensors="pt", padding=True)
            batch_q    = self.processor(text=texts2, images=images, return_tensors="pt", padding=True)

            q_len = batch_q["input_ids"].shape[1]        # length of the question-only prompt
            labels = batch_full["input_ids"].clone()

            # Mask everything before the answer
            labels[:, :q_len] = -100
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            labels[labels == self.image_token_id] = -100

            batch_full["labels"] = labels
            return batch_full

        else:  # inference / validation
            batch_q  = self.processor(text=texts2, images=images, return_tensors="pt", padding=True)
            batch_gt = self.processor(text=texts,  images=images, return_tensors="pt", padding=True)

            q_len = batch_q["input_ids"].shape[1]
            answer_ids = batch_gt["input_ids"][:, q_len:]   # strip the question part

            batch_q["answer"]    = texts                   # raw answers for reference
            batch_q["answer_id"] = answer_ids
            return batch_q

    # ----------------------------------------------------------------------
    def build_message(self, question, answer=None):
        """Format messages for Llava / SmolVLM chat template."""
        messages = [
            {
                "role": "user",
                "content": [
                    # the <image> token is already in the prompt, so no extra token
                    {"type": "text", "text": question}
                ],
            }
        ]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )
        return messages
