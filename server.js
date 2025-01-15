import OpenAI from 'openai'
import { Pinecone } from '@pinecone-database/pinecone'
import 'dotenv/config'

const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
})

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })

const texts = [
    `The artichoke is the finest of all vegetables.`,
    `A nun writing her name in marmalade on a soldier's leg.`,
    'Light coursing from a swarming moon that careens in frozen ecstasy across the sky',
]
async function generateEmbeddings(texts) {
    const embeddingsArray = []
    try {
        for (const text of texts) {
            const response = await openai.embeddings.create({
                model: 'text-embedding-3-large',
                input: text,
                encoding_format: 'float',
            })
            embeddingsArray.push(response.data[0].embedding)
        }
    } catch (error) {
        console.error('Error creating embeddings:', error)
    }
    return embeddingsArray
}

async function createAndStoreEmbeddings(embeddings) {
    const indexName = 'embeddings-index'
    const indexes = await pinecone.listIndexes()

    const doesIndexExist = (name) => {
        console.log('Checking existence of index..', indexes)
        let indexExists = false
        indexes.indexes.forEach((index) => {
            if (index.name === name) {
                indexExists = true
            }
        })
        return indexExists
    }

    if (!doesIndexExist(indexName)) {
        console.log('Building new index...')
        await pinecone.createIndex({
            name: indexName,
            dimension: 3072,
            spec: { serverless: { cloud: 'aws', region: 'us-east-1' } },
        })
        console.log('Index created.')
    }

    const index = await pinecone.Index(indexName)
    const items = [
        { id: 'artichoke', values: embeddings[0] },
        { id: 'marmalade', values: embeddings[1] },
    ]
    await index.upsert(items)
}

async function queryPinecone(queryEmbeddings) {
    const index = pinecone.Index('embeddings-index')
    console.log('Query Embeddings', queryEmbeddings)

    const queryResponse = await index.query({
        // queries: [{ values: queryEmbedding }],
        // topK: 2,
        vector: queryEmbeddings[2],
        topK: 3,
        includeValues: true,
    })

    console.log('Query Results:', queryResponse)
}

generateEmbeddings(texts).then((embeddings) => {
    console.log('Generated Embeddings From OpenAI:', embeddings)
    createAndStoreEmbeddings(embeddings)
    queryPinecone(embeddings)
})
