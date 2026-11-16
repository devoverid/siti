use async_openai::types::chat::ChatCompletionTools;
use serde::Deserialize;
use serde_json::json;

use super::{function, parse_args, ToolContext};
use crate::{agent::Attachment, error::SitiError, meme};

pub const NAME: &str = "add_text_to_image";

pub fn definition() -> ChatCompletionTools {
    function(
        NAME,
        "Make a meme: write caption text on the photo the user sent or replied to, in classic meme style \
         (white uppercase letters with a black outline at the top and/or bottom). The photo itself is kept as is; \
         use this instead of generate_image whenever the user wants text added to their photo. \
         The image is sent to the chat after your reply.",
        json!({
            "type": "object",
            "properties": {
                "top_text": {"type": "string", "description": "Caption at the top of the image."},
                "bottom_text": {"type": "string", "description": "Caption at the bottom of the image."}
            }
        }),
    )
}

#[derive(Deserialize)]
struct Args {
    #[serde(default)]
    top_text: String,
    #[serde(default)]
    bottom_text: String,
}

pub async fn run(args: &str, ctx: &mut ToolContext<'_>) -> Result<String, SitiError> {
    let Args { top_text, bottom_text } = parse_args(args)?;
    let Some(image) = ctx.images.first() else {
        return Err(SitiError::Generate("no image attached; ask the user to send or reply to a photo".into()));
    };
    let bytes = meme::add_text_async(image.bytes.clone(), top_text, bottom_text).await?;
    ctx.attachments.push(Attachment::Image { bytes, caption: String::new() });
    Ok("Meme created; it will be sent to the chat after your reply.".into())
}
