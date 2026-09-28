//! Example 37: TypeSafe System One evaluation with Noul, Choice, and Score.

use superglue::client::ClientBuilder;
use superglue::providers::ProviderId;
use superglue::systemone::{Choice, Noul, Questions, Score, SystemOneRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("TYPESAFE_API_KEY").unwrap_or_default();
    if key.is_empty() {
        eprintln!("Set TYPESAFE_API_KEY to run this example.");
        return Ok(());
    }

    let mut questions = Questions::new();
    questions.insert(
        "is_urgent",
        Noul::new("The message conveys urgency or time-sensitivity"),
    );
    questions.insert(
        "department",
        Choice::new("Which team should handle this")
            .option("billing", "Payment or subscription issues")
            .option("technical", "Bugs or integration problems")
            .option("sales", "Pricing or account questions"),
    );
    questions.insert(
        "frustration",
        Score::new("How frustrated the customer appears").levels([
            "Calm, just stating facts",
            "Frustrated but civil",
            "Very angry, strong language",
        ]),
    );

    let client = ClientBuilder::new()
        .api_key("unused")
        .api_key_for(ProviderId::TypeSafe, key)
        .build()?;

    let response = client
        .system_one(SystemOneRequest::new(
            "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.",
            questions,
        ))
        .await?;

    println!(
        "department: {:?}",
        response.answer("department").and_then(|a| a.choice())
    );
    println!(
        "frustration: {:?}",
        response.answer("frustration").and_then(|a| a.score())
    );
    println!(
        "is_urgent: {:?}",
        response.answer("is_urgent").and_then(|a| a.noul())
    );
    Ok(())
}
