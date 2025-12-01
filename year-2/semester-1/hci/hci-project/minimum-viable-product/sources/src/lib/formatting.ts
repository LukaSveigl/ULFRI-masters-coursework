export function formatDateTime(date: Date, includeTime = true): string {
    const justDate =
        date.getFullYear().toString().padStart(4, "0") +
        "-" +
        (date.getMonth() + 1).toString().padStart(2, "0") +
        "-" +
        date.getDate().toString().padStart(2, "0");
    if (!includeTime) return justDate;
    return (
        justDate +
        " " +
        date.getHours().toString().padStart(2, "0") +
        ":" +
        date.getMinutes().toString().padStart(2, "0") +
        ":" +
        date.getSeconds().toString().padStart(2, "0")
    );
}

export function formatMoney(value: number, forcePrefix = false, eq: string = "+") {
    return `${forcePrefix && value >= 0 ? (value == 0 ? eq : "+") : ""}${value.toFixed(2)} €`;
}
