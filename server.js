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
            // console.log('Embeddings response:', response.data[0].embedding)
            embeddingsArray.push(response.data[0].embedding)
        }
    } catch (error) {
        console.error('Error creating embeddings:', error)
    }
    return embeddingsArray
}

async function createAndStoreEmbeddings(embeddings) {
    // console.log('embeddings:', embeddings)
    const indexName = 'embeddings-index'
    const indexes = await pinecone.listIndexes()
    // console.log('-------------')
    // console.log('Pinecone Indexes:', indexes.indexes)

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
            // dimension: embeddings[0].length, -- OpenAI is returning "undefined", which is breaking it at the moment
            dimension: 3072,
            spec: { serverless: { cloud: 'aws', region: 'us-east-1' } },
        })
        console.log('Index created.')
    }

    const index = await pinecone.Index(indexName)
    const items = [
        { id: 'artichoke', values: embeddings[0] },
        { id: 'marmalade', values: embeddings[1] },
        // { id: 'item_1', values: [0.2, 0.1, 0.3, 0.6, 0.3, 0.7, 0.3, 0.87] },
        // { id: 'item_2', values: [0.01, 0.3, 0.77, 0.65, 0.3, 0.43, 0.2, 0.3] },
    ]
    // console.log('Items from Pinecone function:', items)
    await index.upsert(items)

    // console.log('Embeddings stored in Pinecone!')
}

async function queryPinecone(queryEmbedding) {
    const index = pinecone.Index('embeddings-index')

    const queryResponse = await index.query({
        // queries: [{ values: queryEmbedding }],
        // topK: 2,
        vector: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        topK: 3,
        includeValues: true,
    })

    console.log('Query Results:', queryResponse)
}

generateEmbeddings(texts)
    .then((embeddings) => {
        // console.log('Generated Embeddings From OpenAI:', embeddings)
        createAndStoreEmbeddings(embeddings)
    })
    .then((embeddings) => {
        queryPinecone(embeddings) // embedding values don't matter because everything is hardcoded right now.
    })
