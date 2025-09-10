-- ✅ CORRECTED QUERY: Top 15 providers paid above business average
-- Uses actual payment data from NEGOTIATED_RATES table

WITH PaymentLevels AS (
    SELECT 
        PR."PROVIDER_ID",
        PR."PAYER",
        AVG(NR."NEGOTIATED_AMOUNT") AS AveragePayment
    FROM 
        "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."PROVIDER_REFERENCES" PR
    INNER JOIN 
        "HEALTHCARE_PRICING_ANALYTICS_SAMPLE"."SAMPLES"."NEGOTIATED_RATES" NR
        ON PR."PROVIDER_ID" = NR."PROVIDER_ID"
    GROUP BY 
        PR."PROVIDER_ID", PR."PAYER"
),
BusinessAverage AS (
    SELECT 
        AVG(AveragePayment) AS BookOfBusinessAverage
    FROM 
        PaymentLevels
),
RankedProviders AS (
    SELECT 
        P."PROVIDER_ID",
        P."PAYER",
        P.AveragePayment,
        B.BookOfBusinessAverage,
        P.AveragePayment - B.BookOfBusinessAverage AS PaymentDifference,
        RANK() OVER (ORDER BY P.AveragePayment - B.BookOfBusinessAverage DESC) AS Rank
    FROM 
        PaymentLevels P
    CROSS JOIN 
        BusinessAverage B
    WHERE 
        P.AveragePayment > B.BookOfBusinessAverage  -- Only above average
)

SELECT 
    R."PROVIDER_ID",
    R."PAYER", 
    R.AveragePayment,
    R.BookOfBusinessAverage,
    R.PaymentDifference,
    R.Rank
FROM 
    RankedProviders R
WHERE 
    R.Rank <= 15  -- Top 15 only
ORDER BY 
    R.Rank
LIMIT 15;
